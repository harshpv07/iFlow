const { app, BrowserWindow, ipcMain } = require('electron');

let mainWindow;

app.on('ready', () => {
    mainWindow = new BrowserWindow({
        width: 400, // Reduced width for smaller chat window
        height: 500, // Reduced height for smaller chat window
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            enableRemoteModule: true // Enable remote module for IPC
        },
        show: false, // Hide window until ready
        minWidth: 350, // Reduced minimum window size
        minHeight: 400 // Reduced minimum window size
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
