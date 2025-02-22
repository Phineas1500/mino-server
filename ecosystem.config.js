module.exports = {
  apps: [{
    name: 'mino-server',
    script: 'src/server.js',
    watch: true,
    ignore_watch: [
      'uploads',
      'uploads/*',
      'src/uploads',
      'src/uploads/*',
      'logs/*',
      'node_modules'
    ],
    env: {
      NODE_ENV: 'production',
      PORT: 3001
    }
  }]
}