"use client";

import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader, GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface OrientationData {
  timestamp: number; // Milliseconds relative to action start
  x: number; // Roll
  y: number; // Pitch
  z: number; // Yaw
}

interface LacrosseStickAnimationProps {
  orientationData: OrientationData[];
  modelPath: string; // Path to the .glb model
  isPlaying: boolean; // Controls the animation playback
  actionDurationSeconds: number; // Total duration of the action in seconds
}

const LacrosseStickAnimation: React.FC<LacrosseStickAnimationProps> = ({
  orientationData,
  modelPath,
  isPlaying,
  actionDurationSeconds
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const animationFrameId = useRef<number | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const stickModelRef = useRef<THREE.Group | null>(null);
  const clockRef = useRef(new THREE.Clock());
  const controlsRef = useRef<OrbitControls | null>(null);
  const animationTimeRef = useRef(0);
  const axesHelperRef = useRef<THREE.AxesHelper | null>(null); // Ref for AxesHelper
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0); // Animation progress 0-1
  const [playbackRate, setPlaybackRate] = useState(0.25); // Default speed set to 0.25x

  useEffect(() => {
    if (!mountRef.current || typeof window === 'undefined') return;

    setIsLoading(true);
    setError(null);
    setProgress(0);
    clockRef.current.stop(); // Ensure clock is stopped initially

    // Scene setup
    sceneRef.current = new THREE.Scene();
    sceneRef.current.background = new THREE.Color(0xf0f0f0);

    // Camera setup
    cameraRef.current = new THREE.PerspectiveCamera(50, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 1000);
    cameraRef.current.up.set(0, 0, 1); // Set Z as the up direction for the camera
    cameraRef.current.position.set(100, 100, 100); // Keep your current position, adjust as needed for Z-up view
    cameraRef.current.lookAt(0, 0, 0);

    // Renderer setup
    rendererRef.current = new THREE.WebGLRenderer({ antialias: true });
    rendererRef.current.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    rendererRef.current.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(rendererRef.current.domElement);

    // OrbitControls setup
    if (cameraRef.current && rendererRef.current) {
        controlsRef.current = new OrbitControls(cameraRef.current, rendererRef.current.domElement);
        controlsRef.current.target.set(0, 0, 0); // Orbit around the origin
        controlsRef.current.enableDamping = true;
        controlsRef.current.dampingFactor = 0.1;
        // controlsRef.current.minDistance = 1; // Optional: prevent zooming too close
        // controlsRef.current.maxDistance = 10; // Optional: prevent zooming too far
    }

    // Add AxesHelper
    if (sceneRef.current) {
        axesHelperRef.current = new THREE.AxesHelper(50); // Length of axes lines; adjust as needed
        sceneRef.current.add(axesHelperRef.current);
    }

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    sceneRef.current.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 7.5);
    sceneRef.current.add(directionalLight);

    // Load GLB model
    const loader = new GLTFLoader();
    loader.load(
      modelPath,
      (gltf: GLTF) => {
        console.log('GLTF loaded successfully:', gltf);

        // If a stick model from a previous load exists, remove it first.
        if (stickModelRef.current && sceneRef.current && stickModelRef.current.parent === sceneRef.current) {
            sceneRef.current.remove(stickModelRef.current);
            // Dispose of its geometries and materials if not handled by general scene cleanup
            stickModelRef.current.traverse(object => {
                if (object instanceof THREE.Mesh) {
                    object.geometry.dispose();
                    if (Array.isArray(object.material)) {
                        object.material.forEach(material => material.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            });
        }

        stickModelRef.current = gltf.scene;

        const box = new THREE.Box3().setFromObject(stickModelRef.current);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        console.log('Loaded Model Bounding Box Size:', size);
        console.log('Loaded Model Bounding Box Center:', center);

        // Removed verbose traversal log for all children

        if (stickModelRef.current) {
          stickModelRef.current.scale.set(1, 1, 1); 
          stickModelRef.current.position.set(0, 0, 0);
          sceneRef.current?.add(stickModelRef.current);
        }

        // --- Test Cube Removed ---
        // if (sceneRef.current) {
        //     const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
        //     const material = new THREE.MeshStandardMaterial({ color: 0x00ff00 });
        //     const cube = new THREE.Mesh(geometry, material);
        //     cube.position.set(0, 0.5, 0);
        //     sceneRef.current.add(cube);
        //     console.log('Test cube added to scene.');
        // }
        // --- End Test Cube Removal ---

        setIsLoading(false);
        if (isPlaying) clockRef.current.start();
      },
      undefined,
      (error: unknown) => {
        console.error('Error loading GLB model:', error);
        let message = 'Unknown error';
        if (error instanceof Error) {
          message = error.message;
        } else if (typeof error === 'string') {
          message = error;
        }
        setError('Failed to load 3D model. Path: ' + modelPath + '. Details: ' + message);
        setIsLoading(false);
      }
    );
    
    const currentMount = mountRef.current; // Capture current mountRef

    // Handle resize
    const handleResize = () => {
      if (cameraRef.current && rendererRef.current && currentMount) {
        cameraRef.current.aspect = currentMount.clientWidth / currentMount.clientHeight;
        cameraRef.current.updateProjectionMatrix();
        rendererRef.current.setSize(currentMount.clientWidth, currentMount.clientHeight);
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
      if (rendererRef.current && currentMount) {
         // Check if domElement is a child of currentMount before removing
        if (currentMount.contains(rendererRef.current.domElement)) {
            currentMount.removeChild(rendererRef.current.domElement);
        }
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
      if (sceneRef.current) {
        // Explicitly remove and clean up the loaded stick model and axes helper
        if (stickModelRef.current && stickModelRef.current.parent === sceneRef.current) {
            sceneRef.current.remove(stickModelRef.current);
        }
        if (axesHelperRef.current && axesHelperRef.current.parent === sceneRef.current) {
            sceneRef.current.remove(axesHelperRef.current);
            axesHelperRef.current.dispose(); // Dispose geometry/material of helper
        }
        // Dispose geometries and materials
        sceneRef.current.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            if (object.geometry) object.geometry.dispose();
            if (object.material) {
              if (Array.isArray(object.material)) {
                object.material.forEach(material => material.dispose());
              } else {
                object.material.dispose();
              }
            }
          }
        });
      }
      stickModelRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      rendererRef.current = null;
      clockRef.current.stop();
      controlsRef.current?.dispose();
    };
  }, [modelPath]); // Re-run effect if modelPath changes

  useEffect(() => {
    if (isPlaying && !isLoading && !error) {
        // If animation had ended (or is beyond) and user clicks play again, reset time and progress
        if (animationTimeRef.current >= actionDurationSeconds && actionDurationSeconds > 0) {
            animationTimeRef.current = 0;
            setProgress(0);
        }
        // Ensure clock is running if it should be
        if (!clockRef.current.running) {
            clockRef.current.start(); 
        }
    } else {
        // Pause the clock if not playing, or if loading/error
        clockRef.current.stop(); 
    }
  }, [isPlaying, isLoading, error, actionDurationSeconds]); 

  useEffect(() => {
    if (!orientationData || orientationData.length === 0 || !stickModelRef.current || !sceneRef.current || !cameraRef.current || !rendererRef.current || isLoading || error) {
      if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
      return;
    }

    let lastTimestamp = -1; 

    const animate = () => {
      animationFrameId.current = requestAnimationFrame(animate);
      if (!stickModelRef.current || !sceneRef.current || !cameraRef.current || !rendererRef.current) return;

      controlsRef.current?.update();

      const delta = clockRef.current.getDelta();
      if (isPlaying) {
        animationTimeRef.current += delta * playbackRate;
      }
      // Ensure animationTime does not exceed duration if playing
      if (animationTimeRef.current > actionDurationSeconds && actionDurationSeconds > 0 && isPlaying) {
          animationTimeRef.current = actionDurationSeconds;
          // Consider stopping clock or isPlaying here if we want auto-stop at end
      }

      const currentAnimationTimeS = animationTimeRef.current;
      
      if (isPlaying) { 
          if (actionDurationSeconds > 0) {
              const newSliderProgress = Math.min(currentAnimationTimeS / actionDurationSeconds, 1);
              setProgress(newSliderProgress); 
              
              if (newSliderProgress >= 1) { 
                  // clockRef.current.stop(); // Keep clock running for delta, manage isPlaying via parent
              }
          } else {
              setProgress(0); 
          }
      }

      const currentAnimationTimeMs = currentAnimationTimeS * 1000;

      let prevDataPoint: OrientationData | null = null;
      let nextDataPoint: OrientationData | null = null;

      for (let i = 0; i < orientationData.length; i++) {
        if (orientationData[i].timestamp <= currentAnimationTimeMs) {
          prevDataPoint = orientationData[i];
          if (i + 1 < orientationData.length) {
            nextDataPoint = orientationData[i+1];
          } else {
            nextDataPoint = null; // At or past the last point
          }
        } else {
          // This means currentAnimationTimeMs is before orientationData[i].timestamp
          // If prevDataPoint is still null here, it means we are before the very first data point.
          if (!prevDataPoint) {
             nextDataPoint = orientationData[i]; // The first point becomes the next point to aim for
          }
          break; 
        }
      }
      
      const qTarget = new THREE.Quaternion();

      if (prevDataPoint && nextDataPoint) {
        const sensorRollPrev = prevDataPoint.x * (Math.PI / 180);
        const sensorPitchPrev = prevDataPoint.y * (Math.PI / 180);
        const sensorYawPrev = prevDataPoint.z * (Math.PI / 180);
        // Using Euler order ZYX (Yaw, Pitch, Roll) - common for this type of RPY data
        const qPrev = new THREE.Quaternion().setFromEuler(new THREE.Euler(sensorRollPrev, sensorPitchPrev, sensorYawPrev, 'ZYX'));

        const sensorRollNext = nextDataPoint.x * (Math.PI / 180);
        const sensorPitchNext = nextDataPoint.y * (Math.PI / 180);
        const sensorYawNext = nextDataPoint.z * (Math.PI / 180);
        const qNext = new THREE.Quaternion().setFromEuler(new THREE.Euler(sensorRollNext, sensorPitchNext, sensorYawNext, 'ZYX'));

        // Ensure SLERP takes the shortest path
        if (qPrev.dot(qNext) < 0) {
            qNext.conjugate(); // Or qNext.multiplyScalar(-1); both achieve inversion for this check
        }

        const timeDiff = nextDataPoint.timestamp - prevDataPoint.timestamp;
        let alpha = 0;
        if (timeDiff > 0) {
            alpha = (currentAnimationTimeMs - prevDataPoint.timestamp) / timeDiff;
        }
        alpha = Math.max(0, Math.min(1, alpha)); // Clamp alpha to [0, 1]

        qTarget.copy(qPrev).slerp(qNext, alpha);
        if (stickModelRef.current) stickModelRef.current.quaternion.copy(qTarget);
        lastTimestamp = prevDataPoint.timestamp + alpha * timeDiff; // Approximate timestamp for debugging/last update

      } else if (prevDataPoint) { // Only prevDataPoint exists (at or past the end)
        const sensorRoll = prevDataPoint.x * (Math.PI / 180);
        const sensorPitch = prevDataPoint.y * (Math.PI / 180);
        const sensorYaw = prevDataPoint.z * (Math.PI / 180);
        qTarget.setFromEuler(new THREE.Euler(sensorRoll, sensorPitch, sensorYaw, 'ZYX'));
        if (stickModelRef.current) stickModelRef.current.quaternion.copy(qTarget);
        lastTimestamp = prevDataPoint.timestamp;

      } else if (nextDataPoint) { // Before the first data point, use the first one's orientation
        const sensorRoll = nextDataPoint.x * (Math.PI / 180);
        const sensorPitch = nextDataPoint.y * (Math.PI / 180);
        const sensorYaw = nextDataPoint.z * (Math.PI / 180);
        qTarget.setFromEuler(new THREE.Euler(sensorRoll, sensorPitch, sensorYaw, 'ZYX'));
        if (stickModelRef.current) stickModelRef.current.quaternion.copy(qTarget);
        lastTimestamp = nextDataPoint.timestamp; // Or consider it 0 for interpolation start?
      }
      // No specific update to lastTimestamp if no points - though this case should be handled by the initial guard

      if (stickModelRef.current && rendererRef.current && sceneRef.current && cameraRef.current) {
         rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };

    if (!isLoading && !error) {
        lastTimestamp = -1; 
        if (isPlaying) {
            // If isPlaying is true, animationTimeRef could have been reset by the effect above.
            // Sync progress to the current animationTimeRef.
            if (actionDurationSeconds > 0) {
                setProgress(Math.min(animationTimeRef.current / actionDurationSeconds, 1));
            } else {
                setProgress(0);
            }
            // Ensure clock is running (might have been stopped if just unpaused or started)
            if (!clockRef.current.running) clockRef.current.start();
        } else { // When paused, animationTimeRef should reflect the scrubbed progress
            animationTimeRef.current = progress * actionDurationSeconds;
        }
        
        animate(); 
    }

    return () => {
      if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
    };
  }, [orientationData, isLoading, error, isPlaying, actionDurationSeconds, playbackRate]); 

  const handlePlaybackRateChange = () => {
    setPlaybackRate(currentRate => {
        if (currentRate === 1.0) return 0.5;
        if (currentRate === 0.5) return 0.25;
        if (currentRate === 0.25) return 1.0; // Cycle from 0.25x to 1.0x
        return 0.25; // Default fallback
    });
  };

  if (error) {
    return <div className="text-red-500 p-4 text-sm bg-red-50 rounded-md">Error: {error}</div>;
  }
  
  const canvasHeight = '600px'; // Increased height from 200px to 300px

  return (
    <div className="relative w-full" style={{ height: canvasHeight }}>
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-75 z-10">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <p className="ml-2 text-sm text-gray-600">Loading 3D Model...</p>
        </div>
      )}
      <div ref={mountRef} className="w-full h-full" />
      {!isLoading && !error && orientationData && orientationData.length > 0 && actionDurationSeconds > 0 && (
         <div className="absolute bottom-2 left-2 right-2 px-2">
            <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={progress}
                onChange={(e) => {
                    const newProgress = parseFloat(e.target.value);
                    setProgress(newProgress);
                    if (!isPlaying) { 
                        const newScrubTimeS = newProgress * actionDurationSeconds;
                        animationTimeRef.current = newScrubTimeS; // Update custom animation time
                        // clockRef.current.elapsedTime = newScrubTimeS; // No longer directly setting clock.elapsedTime for scrubbing
                        setProgress(newProgress); 

                        // Force re-render of model at this new scrubbed position
                         const currentAnimationTimeMs = newScrubTimeS * 1000;
                         let pointToApply: OrientationData | null = null;
                         // Find the closest actual data point to the scrub time, no interpolation for scrubbing preview
                         if (orientationData.length > 0) {
                            pointToApply = orientationData[0]; // Default to first
                            for (const point of orientationData) {
                                if (point.timestamp <= currentAnimationTimeMs) {
                                    pointToApply = point;
                                } else {
                                    // If current time is closer to next point, consider that too for snapping?
                                    // For now, just use the last one we passed.
                                    break;
                                }
                            }
                         }

                         if (pointToApply && stickModelRef.current) {
                            const sensorRoll = pointToApply.x * (Math.PI / 180);
                            const sensorPitch = pointToApply.y * (Math.PI / 180);
                            const sensorYaw = pointToApply.z * (Math.PI / 180);

                            const euler = new THREE.Euler(sensorRoll, sensorPitch, sensorYaw, 'ZYX');
                            const quaternion = new THREE.Quaternion().setFromEuler(euler);
                            stickModelRef.current.quaternion.copy(quaternion);
                         } else if (orientationData.length > 0 && stickModelRef.current) { // Default to first if no prior
                            const firstPoint = orientationData[0];
                            const sensorRoll = firstPoint.x * (Math.PI / 180);
                            const sensorPitch = firstPoint.y * (Math.PI / 180);
                            const sensorYaw = firstPoint.z * (Math.PI / 180);

                            const euler = new THREE.Euler(sensorRoll, sensorPitch, sensorYaw, 'ZYX');
                            const quaternion = new THREE.Quaternion().setFromEuler(euler);
                            stickModelRef.current.quaternion.copy(quaternion);
                         }
                         if(rendererRef.current && sceneRef.current && cameraRef.current){
                            rendererRef.current.render(sceneRef.current, cameraRef.current);
                         }
                    }
                }}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                disabled={isPlaying}
            />
        </div>
      )}
      {!isLoading && !error && (
        <div className="absolute top-2 right-2">
            <button 
                onClick={handlePlaybackRateChange} 
                className="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded"
            >
                Speed: {playbackRate}x
            </button>
        </div>
      )}
    </div>
  );
};

export { LacrosseStickAnimation }; 