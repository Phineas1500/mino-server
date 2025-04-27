require('dotenv').config();
module.exports = {
  apps: [{
    name: 'mino-server',
    script: 'src/server.js',
    watch: true,
    ignore_watch: [
      'uploads',
      'uploads/**',
      'src/uploads',
      'src/uploads/**',
      'temp',
      'temp/**',
      'src/temp',
      'src/temp/**',
      'logs',
      'logs/**',
      'node_modules',
      '/tmp',
      '*.mp4',
      '*.wav',
      '*.mov',
      '*TEMP*',
      '**/*TEMP*',
      '**/tmp*'
    ],
    watch_options: {
      followSymlinks: false,
      usePolling: true,
      alwaysStat: false,
      ignoreInitial: true
    },
    env: {
      NODE_ENV: 'production',
      PORT: process.env.PORT
    }
  }]
}
