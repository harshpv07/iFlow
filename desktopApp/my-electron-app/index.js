const { app, BrowserWindow, ipcMain, globalShortcut } = require('electron');

let mainWindow;

app.on('ready', () => {
    mainWindow = new BrowserWindow({
        width: 600,
        height: 60,
        frame: false,
        transparent: true,
        alwaysOnTop: true,
        visibleOnAllWorkspaces: true,
        hasShadow: false,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            enableRemoteModule: true
        },
        show: false,
        x: 0, // Will be updated to position window in top right
        y: 0,
        resizable: false,
        scrollable: false
    });

    mainWindow.loadFile('index.html');
    
    // Position window in top right corner
    const { screen } = require('electron');
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width: screenWidth } = primaryDisplay.workAreaSize;
    mainWindow.setPosition(screenWidth - 620, 10);

    // Show window when content is ready
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
    });

    // Register global shortcut
    globalShortcut.register('Super+Space', () => {
        if (mainWindow.isVisible()) {
            mainWindow.hide();
        } else {
            mainWindow.show();
        }
    });

    // Handle window closing 
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
});
