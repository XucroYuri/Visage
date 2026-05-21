use serde::Serialize;
use std::io::BufRead;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::{
    Emitter,
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager,
};

const HEALTH_URL: &str = "http://localhost:8787/api/health";
const SIDECAR_START_TIMEOUT_MS: u64 = 30_000;
const HEALTH_POLL_INTERVAL_MS: u64 = 200;

#[derive(Serialize, Clone)]
struct AppInfo {
    version: String,
    platform: String,
    python_running: bool,
}

struct SidecarState {
    child: Option<Child>,
}

#[tauri::command]
fn get_app_info(state: tauri::State<Mutex<SidecarState>>) -> AppInfo {
    let sidecar = state.lock().unwrap();
    AppInfo {
        version: "0.1.0".to_string(),
        platform: std::env::consts::OS.to_string(),
        python_running: sidecar.child.is_some(),
    }
}

/// Poll health endpoint until backend is ready or timeout.
async fn wait_for_backend() -> Result<(), String> {
    let start = std::time::Instant::now();
    while start.elapsed().as_millis() < SIDECAR_START_TIMEOUT_MS as u128 {
        if let Ok(resp) = reqwest::get(HEALTH_URL).await {
            if resp.status().is_success() {
                return Ok(());
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(HEALTH_POLL_INTERVAL_MS)).await;
    }
    Err("Backend did not start within 30 seconds".to_string())
}

/// Spawn the Python backend as a sidecar process.
fn spawn_sidecar(app_handle: &tauri::AppHandle) -> Option<Child> {
    let python = if cfg!(target_os = "windows") {
        "python"
    } else {
        "python3"
    };

    match Command::new(python)
        .args([
            "-m",
            "visage.server.app",
            "--serve",
            "--port",
            "8787",
            "--no-open",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(mut child) => {
            let handle = app_handle.clone();
            let pid = child.id();
            // Forward stderr to frontend via events
            if let Some(stderr) = child.stderr.take() {
                std::thread::spawn(move || {
                    let mut reader = std::io::BufReader::new(stderr);
                    let mut line = String::new();
                    while reader.read_line(&mut line).is_ok() {
                        if line.is_empty() {
                            break;
                        }
                        let _ = handle.emit("sidecar-log", line.trim());
                        line.clear();
                    }
                });
            }
            log::info!("Sidecar started (PID: {})", pid);
            Some(child)
        }
        Err(e) => {
            log::error!("Failed to start sidecar: {}", e);
            None
        }
    }
}

fn build_tray(app: &tauri::App) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, "show", "Show Window", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, Some("CmdOrCtrl+Q"))?;
    let menu = Menu::with_items(app, &[&show, &quit])?;

    let _tray = TrayIconBuilder::new()
        .icon(app.default_window_icon().unwrap().clone())
        .menu(&menu)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            "quit" => {
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
        })
        .build(app)?;

    Ok(())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .manage(Mutex::new(SidecarState {
            child: None,
        }))
        .invoke_handler(tauri::generate_handler![get_app_info])
        .setup(|app| {
            build_tray(app)?;

            let handle = app.handle().clone();
            let state = app.state::<Mutex<SidecarState>>();

            // Spawn Python engine sidecar
            let child = spawn_sidecar(&handle);
            if let Some(c) = child {
                let mut sidecar = state.lock().unwrap();
                sidecar.child = Some(c);
            }

            // Wait for backend in background
            let handle_clone = handle.clone();
            tauri::async_runtime::spawn(async move {
                match wait_for_backend().await {
                    Ok(()) => {
                        log::info!("Backend is ready");
                        let _ = handle_clone.emit("backend-ready", ());
                    }
                    Err(e) => {
                        log::error!("Backend startup failed: {}", e);
                        let _ = handle_clone.emit("backend-error", e);
                    }
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // Hide instead of closing (keep running in tray)
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Visage");
}
