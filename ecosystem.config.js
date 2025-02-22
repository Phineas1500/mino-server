// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'mino-server',
    script: 'src/server.js',
    watch: true,
    ignore_watch: ['uploads/*', 'node_modules'],
    env: {
      NODE_ENV: 'production',
      PORT: 3001
    },
    error_file: 'logs/error.log',
    out_file: 'logs/output.log',
    time: true
  }]
}
