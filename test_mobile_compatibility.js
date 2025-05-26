#!/usr/bin/env node

/**
 * Mobile Video Compatibility Test Script
 * 
 * Usage:
 * node test_mobile_compatibility.js check <video_file_path>
 * node test_mobile_compatibility.js download <youtube_url>
 * node test_mobile_compatibility.js optimize <input_file> <output_file>
 */

const path = require('path');
const { spawn } = require('child_process');
const mobileOptimizer = require('./src/mobile_video_optimizer');

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  try {
    switch (command) {
      case 'check':
        if (!args[1]) {
          console.error('Usage: node test_mobile_compatibility.js check <video_file_path>');
          process.exit(1);
        }
        await checkVideoCompatibility(args[1]);
        break;

      case 'download':
        if (!args[1]) {
          console.error('Usage: node test_mobile_compatibility.js download <youtube_url>');
          process.exit(1);
        }
        await downloadMobileOptimized(args[1]);
        break;

      case 'optimize':
        if (!args[1] || !args[2]) {
          console.error('Usage: node test_mobile_compatibility.js optimize <input_file> <output_file>');
          process.exit(1);
        }
        await optimizeExistingVideo(args[1], args[2]);
        break;

      case 'formats':
        if (!args[1]) {
          console.error('Usage: node test_mobile_compatibility.js formats <youtube_url>');
          process.exit(1);
        }
        await showAvailableFormats(args[1]);
        break;

      case 'manual-merge':
        if (!args[1] || !args[2] || !args[3]) {
          console.error('Usage: node test_mobile_compatibility.js manual-merge <video_file> <audio_file> <output_file>');
          process.exit(1);
        }
        await manualMergeForMobile(args[1], args[2], args[3]);
        break;

      case 're-encode':
        if (!args[1] || !args[2] || !args[3]) {
          console.error('Usage: node test_mobile_compatibility.js re-encode <video_file> <audio_file> <output_file>');
          process.exit(1);
        }
        await reencodeForMobileCompatibility(args[1], args[2], args[3]);
        break;

      default:
        console.log('Available commands:');
        console.log('  check <video_file>     - Check if video is mobile-compatible');
        console.log('  download <youtube_url> - Download video with mobile optimization');
        console.log('  optimize <input> <output> - Re-encode existing video for mobile');
        console.log('  formats <youtube_url>  - Show available formats and mobile recommendation');
        console.log('  manual-merge <video_file> <audio_file> <output_file> - Manually merge video and audio files (fast)');
        console.log('  re-encode <video_file> <audio_file> <output_file> - Re-encode for maximum mobile compatibility');
        process.exit(1);
    }
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

async function checkVideoCompatibility(filePath) {
  console.log(`\n🔍 Checking mobile compatibility for: ${filePath}\n`);
  
  const analysis = await mobileOptimizer.checkMobileCompatibility(filePath);
  
  console.log(`✅ Mobile Compatible: ${analysis.isCompatible ? 'YES' : 'NO'}\n`);
  
  if (analysis.details.container) {
    console.log('📦 Container:');
    console.log(`   Format: ${analysis.details.container.format}`);
    console.log(`   Mobile Compatible: ${analysis.details.container.compatible ? 'YES' : 'NO'}\n`);
  }
  
  if (analysis.details.video) {
    console.log('🎥 Video:');
    console.log(`   Codec: ${analysis.details.video.codec}`);
    console.log(`   Profile: ${analysis.details.video.profile}`);
    console.log(`   Level: ${analysis.details.video.level}`);
    console.log(`   Resolution: ${analysis.details.video.width}x${analysis.details.video.height}`);
    console.log(`   Bitrate: ${Math.round(analysis.details.video.bitRate / 1000)} kbps\n`);
  }
  
  if (analysis.details.audio) {
    console.log('🔊 Audio:');
    console.log(`   Codec: ${analysis.details.audio.codec}`);
    console.log(`   Sample Rate: ${analysis.details.audio.sampleRate} Hz`);
    console.log(`   Channels: ${analysis.details.audio.channels}`);
    console.log(`   Bitrate: ${Math.round(analysis.details.audio.bitRate / 1000)} kbps\n`);
  }
  
  if (analysis.issues.length > 0) {
    console.log('⚠️  Issues Found:');
    analysis.issues.forEach(issue => console.log(`   • ${issue}`));
    console.log('');
  }
  
  if (analysis.recommendations.length > 0) {
    console.log('💡 Recommendations:');
    analysis.recommendations.forEach(rec => console.log(`   • ${rec}`));
    console.log('');
  }
}

async function downloadMobileOptimized(url) {
  console.log(`\n📱 Downloading mobile-optimized video from: ${url}\n`);
  
  // Get optimal format for mobile
  const optimalFormat = await mobileOptimizer.getOptimalMobileFormat(url);
  console.log(`📋 Using format: ${optimalFormat}\n`);
  
  // First, try to download without merging to avoid hanging
  const ytdlpArgs = [
    '--format', optimalFormat,
    '--keep-video',  // Keep separate video/audio files
    '--no-post-overwrites',
    '--prefer-ffmpeg',
    '--no-warnings',
    '--progress',
    '--abort-on-unavailable-fragment',  // Abort on stream issues
    '--fragment-retries', '3',          // Limit retries
    '--socket-timeout', '30',           // Socket timeout
    '--', url
  ];
  
  console.log('🚀 Starting download (without auto-merge)...');
  console.log(`Command: yt-dlp ${ytdlpArgs.join(' ')}\n`);
  
  const ytdlp = spawn('yt-dlp', ytdlpArgs, { stdio: 'inherit' });
  
  return new Promise((resolve, reject) => {
    // Set a shorter timeout for download only (3 minutes)
    const timeout = setTimeout(() => {
      console.log('\n⏰ Download timeout reached, terminating process...');
      ytdlp.kill('SIGTERM');
      
      // If SIGTERM doesn't work, force kill after 5 seconds
      setTimeout(() => {
        if (!ytdlp.killed) {
          console.log('🔨 Force killing hung process...');
          ytdlp.kill('SIGKILL');
        }
      }, 5000);
      
      reject(new Error('Download timed out after 3 minutes'));
    }, 3 * 60 * 1000); // 3 minutes timeout for download
    
    ytdlp.on('close', (code) => {
      clearTimeout(timeout);
      if (code === 0) {
        console.log('\n✅ Download completed successfully!');
        console.log('📝 Note: Video and audio may be in separate files. Use ffmpeg to merge manually if needed.');
        resolve();
      } else {
        reject(new Error(`yt-dlp failed with code ${code}`));
      }
    });
    
    ytdlp.on('error', (error) => {
      clearTimeout(timeout);
      reject(new Error(`Failed to start yt-dlp: ${error.message}`));
    });
  });
}

async function optimizeExistingVideo(inputPath, outputPath) {
  console.log(`\n🔧 Optimizing video for mobile compatibility...\n`);
  console.log(`Input: ${inputPath}`);
  console.log(`Output: ${outputPath}\n`);
  
  // First check current compatibility
  console.log('📋 Analyzing current video...');
  const analysis = await mobileOptimizer.checkMobileCompatibility(inputPath);
  
  if (analysis.isCompatible) {
    console.log('✅ Video is already mobile-compatible!');
    console.log('   You may still want to optimize for file size or specific requirements.\n');
  } else {
    console.log('⚠️  Video needs optimization for mobile compatibility.\n');
  }
  
  console.log('🚀 Starting optimization...');
  await mobileOptimizer.optimizeForMobile(inputPath, outputPath);
  
  console.log('\n📋 Verifying optimized video...');
  const newAnalysis = await mobileOptimizer.checkMobileCompatibility(outputPath);
  
  if (newAnalysis.isCompatible) {
    console.log('✅ Optimization successful! Video is now mobile-compatible.');
  } else {
    console.log('⚠️  Optimization completed, but some compatibility issues may remain.');
    console.log('Issues:', newAnalysis.issues.join(', '));
  }
}

async function showAvailableFormats(url) {
  console.log(`\n📋 Available formats for: ${url}\n`);
  
  const ytdlp = spawn('yt-dlp', ['-F', '--', url]);
  
  let output = '';
  
  ytdlp.stdout.on('data', (data) => {
    output += data.toString();
    process.stdout.write(data);
  });
  
  ytdlp.stderr.on('data', (data) => {
    process.stderr.write(data);
  });
  
  return new Promise(async (resolve, reject) => {
    ytdlp.on('close', async (code) => {
      if (code !== 0) {
        return reject(new Error('yt-dlp failed'));
      }
      
      try {
        const optimalFormat = await mobileOptimizer.getOptimalMobileFormat(url);
        console.log(`\n💡 Recommended mobile format: ${optimalFormat}`);
        resolve();
      } catch (error) {
        console.log('\n⚠️  Could not determine optimal mobile format');
        resolve();
      }
    });
    
    ytdlp.on('error', (error) => {
      reject(new Error(`Failed to start yt-dlp: ${error.message}`));
    });
  });
}

async function manualMergeForMobile(videoFile, audioFile, outputFile) {
  console.log(`\n🔧 Manually merging (fast copy mode)...`);
  console.log(`Video: ${videoFile}`);
  console.log(`Audio: ${audioFile}`);
  console.log(`Output: ${outputFile}\n`);
  
  const ffmpegArgs = [
    '-i', videoFile,
    '-i', audioFile,
    '-c:v', 'copy',  // Copy video stream to avoid re-encoding
    '-c:a', 'aac',
    '-b:a', '128k',
    '-ar', '44100',
    '-ac', '2',
    '-movflags', '+faststart',
    '-y', // Overwrite output file
    outputFile
  ];
  
  console.log('🚀 Starting manual merge (copy video, re-encode audio)...');
  console.log(`Command: ffmpeg ${ffmpegArgs.join(' ')}\n`);
  
  const ffmpeg = spawn('ffmpeg', ffmpegArgs, { stdio: 'inherit' });
  
  return new Promise((resolve, reject) => {
    // Set a longer timeout for merging (5 minutes)
    const timeout = setTimeout(() => {
      console.log('\n⏰ Merge timeout reached, terminating process...');
      ffmpeg.kill('SIGTERM');
      
      setTimeout(() => {
        if (!ffmpeg.killed) {
          console.log('🔨 Force killing hung merge process...');
          ffmpeg.kill('SIGKILL');
        }
      }, 5000);
      
      reject(new Error('Merge timed out after 5 minutes'));
    }, 5 * 60 * 1000); // 5 minutes timeout
    
    ffmpeg.on('close', (code) => {
      clearTimeout(timeout);
      if (code === 0) {
        console.log('\n✅ Manual merge completed successfully!');
        resolve();
      } else {
        reject(new Error(`ffmpeg failed with code ${code}`));
      }
    });
    
    ffmpeg.on('error', (error) => {
      clearTimeout(timeout);
      reject(new Error(`Failed to start ffmpeg: ${error.message}`));
    });
  });
}

async function reencodeForMobileCompatibility(videoFile, audioFile, outputFile) {
  console.log(`\n🔧 Re-encoding for maximum mobile compatibility...`);
  console.log(`Video: ${videoFile}`);
  console.log(`Audio: ${audioFile}`);
  console.log(`Output: ${outputFile}\n`);
  
  const ffmpegArgs = [
    '-i', videoFile,
    '-i', audioFile,
    '-c:v', 'libx264',
    '-profile:v', 'main',
    '-level:v', '3.1',
    '-preset', 'medium',
    '-crf', '23',
    '-maxrate', '2000k',
    '-bufsize', '4000k',
    '-c:a', 'aac',
    '-b:a', '128k',
    '-ar', '44100',
    '-ac', '2',
    '-movflags', '+faststart',
    '-y', // Overwrite output file
    outputFile
  ];
  
  console.log('🚀 Starting mobile re-encoding (this will take longer)...');
  console.log(`Command: ffmpeg ${ffmpegArgs.join(' ')}\n`);
  
  const ffmpeg = spawn('ffmpeg', ffmpegArgs, { stdio: 'inherit' });
  
  return new Promise((resolve, reject) => {
    // Set a very long timeout for re-encoding (15 minutes)
    const timeout = setTimeout(() => {
      console.log('\n⏰ Re-encoding timeout reached, terminating process...');
      ffmpeg.kill('SIGTERM');
      
      setTimeout(() => {
        if (!ffmpeg.killed) {
          console.log('🔨 Force killing hung re-encoding process...');
          ffmpeg.kill('SIGKILL');
        }
      }, 10000);
      
      reject(new Error('Re-encoding timed out after 15 minutes'));
    }, 15 * 60 * 1000); // 15 minutes timeout
    
    ffmpeg.on('close', (code) => {
      clearTimeout(timeout);
      if (code === 0) {
        console.log('\n✅ Mobile re-encoding completed successfully!');
        resolve();
      } else {
        reject(new Error(`ffmpeg failed with code ${code}`));
      }
    });
    
    ffmpeg.on('error', (error) => {
      clearTimeout(timeout);
      reject(new Error(`Failed to start ffmpeg: ${error.message}`));
    });
  });
}

if (require.main === module) {
  main();
}

module.exports = {
  checkVideoCompatibility,
  downloadMobileOptimized,
  optimizeExistingVideo,
  showAvailableFormats,
  manualMergeForMobile,
  reencodeForMobileCompatibility
}; 