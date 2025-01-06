const { app, BrowserWindow, ipcMain } = require('electron');

let mainWindow;

app.on('ready', () => {
    mainWindow = new BrowserWindow({
        width: 900, // Adjusted for chat interface width
        height: 700, // Adjusted for chat interface height
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            enableRemoteModule: true // Enable remote module for IPC
        },
        show: false, // Hide window until ready
        minWidth: 850, // Set minimum window size
        minHeight: 600
    });

    mainWindow.loadFile('index.html');
    
    // Show window when content is ready
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        mainWindow.center(); // Center the window on screen
    });

    // Handle window closing
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
});
