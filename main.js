import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GPUComputationRenderer } from 'three/addons/misc/GPUComputationRenderer.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { FXAAShader } from 'three/addons/shaders/FXAAShader.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';

// --- Configuration ---
const WIDTH = 128; // Texture width for simulation (total birds = WIDTH * WIDTH)
const NUM_BIRDS = WIDTH * WIDTH;
const BOUNDS = 800; // Simulation area size
const BOUNDS_HALF = BOUNDS / 2;

const MAX_FOOD = 30;       // Max number of food sources
const MAX_PREDATORS = 16;  // Max number of predators
const FOOD_LIFETIME = 60.0; // Seconds before food relocates
const PREDATOR_LIFETIME = 20.0; // Seconds before predator relocates

const BirdGeometry = new THREE.ConeGeometry(1, 4, 6); // Simple bird shape
BirdGeometry.rotateX(Math.PI / 2); // Point the cone forward

// --- Boid Simulation Parameters (will be adjustable via GUI) ---
const PARAMS = {
    separationDistance: 10.0,
    alignmentDistance: 40.0,
    cohesionDistance: 40.0,
    freedomFactor: 0.01,
    // predator: new THREE.Vector3(), // REMOVE (replaced by arrays)
    // hasPredator: false,          // REMOVE

    // Weights
    separationWeight: 0.4,
    alignmentWeight: 4.0,
    cohesionWeight: 5.0,
    predatorWeight: 0.1,   // Weight for automatic predators
    foodWeight: 1.0,       // Weight for food attraction

    // Influence Radii
    predatorRadius: 70.0,  // Radius within which predators repel
    foodRadius: 150.0,      // Radius within which food attracts
    foodOrbitSpeed: 0.2, // Base speed factor for food orbits

    // Limits
    maxSpeed: 100.0,
    maxSteerForce: 25,

    // Visuals & Simulation... (keep existing)
    birdScale: 1.2,
    bloomThreshold: 0.3,
    bloomStrength: 0.6,
    bloomRadius: 0.1,
    timeScale: 1.0,
};

// --- Global Variables ---
let scene, camera, renderer, controls;
let gpuCompute;
let velocityVariable, positionVariable;
let positionUniforms, velocityUniforms;
let birdMesh;
let composer, bloomPass, fxaaPass;
let stats; // Optional: For performance monitoring (add three/addons/libs/stats.module.js if needed)
let clock = new THREE.Clock();
let gui;
let mouse = new THREE.Vector2(-1, -1); // Normalized mouse coords
let raycaster = new THREE.Raycaster();
let targetMesh; // Invisible mesh for raycasting mouse position

let foodSources = [];      // Array to hold { position, orbitParams:{radius, speedFactor, phase, vertAmp, vertFreq, vertPhase}, active, visual }
let predatorSources = [];  // Array to hold { position: Vector3, timer: float, active: bool, visual: Mesh }

// --- Shader Definitions ---

// Fragment shader for computing velocity
const velocityShader = /* glsl */`
    #define MAX_FOOD ${MAX_FOOD}
    #define MAX_PREDATORS ${MAX_PREDATORS}

    uniform float time;
    uniform float delta;
    uniform float separationDistance;
    uniform float alignmentDistance;
    uniform float cohesionDistance;
    uniform float freedomFactor;
    // uniform vec3 predator; // REMOVE
    // uniform bool hasPredator; // REMOVE

    // NEW Uniforms for automatic sources
    uniform vec3 foodPositions[MAX_FOOD];
    uniform bool foodActive[MAX_FOOD];
    uniform vec3 predatorPositions[MAX_PREDATORS];
    uniform bool predatorActive[MAX_PREDATORS];

    // Weights
    uniform float separationWeight;
    uniform float alignmentWeight;
    uniform float cohesionWeight;
    uniform float predatorWeight; // Now for automatic predators
    uniform float foodWeight;     // Weight for food attraction

    // Influence Radii
    uniform float predatorRadius;
    uniform float foodRadius;

    // Limits
    uniform float maxSpeed;
    uniform float maxSteerForce;

    const float BOUNDS = ${BOUNDS.toFixed(1)};
    const float BOUNDS_HALF = ${BOUNDS_HALF.toFixed(1)};
    const float PI = 3.14159265359;
    const float MASS = 1.0;

    // Simple pseudo-random function
    float rand(vec2 co){
        return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
    }

    // Limit vector magnitude
    vec3 limit(vec3 vec, float maxVal) {
        if (length(vec) > maxVal) {
            return normalize(vec) * maxVal;
        }
        return vec;
    }

    void main() {
        vec2 uv = gl_FragCoord.xy / resolution.xy;
        vec3 position = texture2D( texturePosition, uv ).xyz;
        vec3 velocity = texture2D( textureVelocity, uv ).xyz;

        // --- Calculate Steering Forces ---
        vec3 separationForce = vec3(0.0);
        vec3 alignmentForce = vec3(0.0);
        vec3 cohesionForce = vec3(0.0);
        vec3 predatorForce = vec3(0.0);

        vec3 avgVelocity = vec3(0.0);
        vec3 avgPosition = vec3(0.0);
        int alignCount = 0;
        int cohesionCount = 0;
        int separationCount = 0;

        // Iterate through neighbors (sampling)
        const float numSamples = 30.0;
        for (float i = 0.0; i < numSamples; i++) {
            vec2 sampleUV = fract(uv + vec2(rand(uv + i*0.1), rand(uv - i*0.1)) * 0.1); // Slightly different sampling
            vec3 neighborPos = texture2D(texturePosition, sampleUV).xyz;
            vec3 neighborVel = texture2D(textureVelocity, sampleUV).xyz;
            vec3 diff = position - neighborPos;
            float dist = length(diff);

            if (dist > 0.0) { // Check distance > 0
                // Separation
                if (dist < separationDistance) {
                    separationForce += normalize(diff) / max(dist, 0.01); // Force stronger when closer
                    separationCount++;
                }
                // Alignment (accumulate velocities)
                if (dist < alignmentDistance) {
                    avgVelocity += neighborVel;
                    alignCount++;
                }
                // Cohesion (accumulate positions)
                if (dist < cohesionDistance) {
                    avgPosition += neighborPos;
                    cohesionCount++;
                }
            }
        }

        vec3 totalSteeringForce = vec3(0.0);

        // Separation Steering
        if (separationCount > 0) {
            separationForce /= float(separationCount);
            if (length(separationForce) > 0.0) {
                 // Calculate desired velocity (maxSpeed away from average separation vector)
                 vec3 desiredSep = normalize(separationForce) * maxSpeed;
                 // Calculate steering force (Desired - Current)
                 separationForce = desiredSep - velocity; // This is now a steering vector
                 totalSteeringForce += separationForce * separationWeight;
            }
        }

        // Alignment Steering
        if (alignCount > 0) {
            avgVelocity /= float(alignCount);
             // Calculate desired velocity (maxSpeed in direction of average velocity)
            vec3 desiredAlign = normalize(avgVelocity) * maxSpeed;
            // Calculate steering force (Desired - Current)
            alignmentForce = desiredAlign - velocity; // Steering vector
            totalSteeringForce += alignmentForce * alignmentWeight;
        }

        // Cohesion Steering
        if (cohesionCount > 0) {
            avgPosition /= float(cohesionCount);
            vec3 vecToCenter = avgPosition - position;
            if(length(vecToCenter) > 0.0) {
                // Calculate desired velocity (maxSpeed towards center of mass)
                vec3 desiredCoh = normalize(vecToCenter) * maxSpeed;
                // Calculate steering force (Desired - Current)
                cohesionForce = desiredCoh - velocity; // Steering vector
                totalSteeringForce += cohesionForce * cohesionWeight;
            }
        }

        // --- Add Steering for Automatic Food Sources (Attractors) ---
        vec3 foodSteeringForce = vec3(0.0);
        int foodCount = 0;
        for (int i = 0; i < MAX_FOOD; i++) {
            if (foodActive[i]) {
                vec3 vecToFood = foodPositions[i] - position;
                float distToFood = length(vecToFood);

                if (distToFood > 0.0 && distToFood < foodRadius) {
                    // Calculate desired velocity (maxSpeed towards food)
                    vec3 desiredFood = normalize(vecToFood) * maxSpeed;
                    // Calculate steering force (Desired - Current)
                    foodSteeringForce += (desiredFood - velocity);
                    foodCount++;
                }
            }
        }
        if (foodCount > 0) {
            foodSteeringForce /= float(foodCount); // Average steering if multiple foods nearby
            totalSteeringForce += foodSteeringForce * foodWeight;
        }


        // --- Add Steering for Automatic Predator Sources (Repulsors) ---
        vec3 predatorSteeringForce = vec3(0.0);
        int predatorCount = 0;
        for (int i = 0; i < MAX_PREDATORS; i++) {
             if (predatorActive[i]) {
                vec3 vecFromPredator = position - predatorPositions[i]; // Vector pointing away
                float distToPredator = length(vecFromPredator);

                if (distToPredator > 0.0 && distToPredator < predatorRadius) {
                    // Calculate desired velocity (maxSpeed directly away)
                    vec3 desiredFlee = normalize(vecFromPredator) * maxSpeed;
                     // Calculate steering force (Desired - Current)
                     // Make force stronger closer to the predator
                    predatorSteeringForce += (desiredFlee - velocity) * (predatorRadius / max(distToPredator, 0.1));
                    predatorCount++;
                }
             }
        }
        if (predatorCount > 0) {
            predatorSteeringForce /= float(predatorCount); // Average steering if multiple predators nearby
            totalSteeringForce += predatorSteeringForce * predatorWeight;
        }


        // --- Apply Random Steering ("freedom") ---
        if (rand(uv + time * 0.1) > 0.6) {
             float angle = (rand(uv + time * 0.2) - 0.5) * 2.0 * PI * freedomFactor * 0.1; // Small random angle change
             // Simple random turn - could be improved
             vec3 randomSteer = vec3(cos(angle), sin(angle), (rand(uv + time * 0.3) - 0.5) * 0.5); // Mostly planar random turn
             // Rotate random steer to align with current velocity direction somewhat? Or just add it?
             // For simplicity, add a small world-space random nudge
             totalSteeringForce += randomSteer * 0.5; // Add small random force
        }


        // --- Apply Total Steering ---
        totalSteeringForce = limit(totalSteeringForce, maxSteerForce);
        vec3 acceleration = totalSteeringForce / MASS;
        velocity += acceleration * delta;
        velocity = limit(velocity, maxSpeed);

        gl_FragColor = vec4( velocity, 1.0 );
    }
`;

// Fragment shader for computing position
const positionShader = /* glsl */`
    uniform float delta; // Added delta time

    const float BOUNDS = ${BOUNDS.toFixed(1)};
    const float BOUNDS_HALF = ${BOUNDS_HALF.toFixed(1)};

    // Function to wrap position around bounds
    vec3 wrapAround(vec3 pos) {
        pos.x = mod(pos.x + BOUNDS_HALF, BOUNDS) - BOUNDS_HALF;
        pos.y = mod(pos.y + BOUNDS_HALF, BOUNDS) - BOUNDS_HALF;
        pos.z = mod(pos.z + BOUNDS_HALF, BOUNDS) - BOUNDS_HALF;
        return pos;
    }


    void main() {
        vec2 uv = gl_FragCoord.xy / resolution.xy;
        vec3 position = texture2D( texturePosition, uv ).xyz;
        vec3 velocity = texture2D( textureVelocity, uv ).xyz;

        // Update position based on velocity and delta time
        position += velocity * delta;

        // Wrap position around the simulation bounds
        position = wrapAround(position);

        gl_FragColor = vec4( position, 1.0 );
    }
`;

// --- Initialization Functions ---

function init() {
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050510, 0.00001); // Add atmospheric fog

    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
    camera.position.z = BOUNDS * 0.6;
    camera.position.y = BOUNDS * 0.2;

    renderer = new THREE.WebGLRenderer({ antialias: false }); // AA handled by post-processing
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.toneMapping = THREE.ACESFilmicToneMapping; // For better HDR handling with Bloom
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    document.body.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 50;
    controls.maxDistance = 1000;
    controls.target.set(0, 0, 0);

    // Invisible plane for raycasting mouse position onto the scene's XY plane
    // const targetGeometry = new THREE.PlaneGeometry(BOUNDS * 2, BOUNDS * 2);
    // const targetMaterial = new THREE.MeshBasicMaterial({ visible: false, side: THREE.DoubleSide });
    // targetMesh = new THREE.Mesh(targetGeometry, targetMaterial);
    // targetMesh.rotation.x = -Math.PI / 2; // Lay flat on XZ plane
    // scene.add(targetMesh);


    initGPUCompute();
    initBirds();
    initLighting();
    initEnvironment();
    initAttractorsPredators();
    // initStarfield(); 
    initPostProcessing();
    initGUI();

    // Event Listeners
    window.addEventListener('resize', onWindowResize); // Keep resize on window
    // renderer.domElement.addEventListener('pointerdown', onPointerDown); // Attach to canvas
    // renderer.domElement.addEventListener('pointerup', onPointerUp);     // Attach to canvas
    // renderer.domElement.addEventListener('pointermove', onPointerMove); // Attach to canvas

    // Optional: Stats
    // stats = new Stats();
    // document.body.appendChild(stats.dom);

    console.log(`Initialized ${NUM_BIRDS} boids.`);
}

function initGPUCompute() {
    gpuCompute = new GPUComputationRenderer(WIDTH, WIDTH, renderer);

    if (renderer.capabilities.isWebGL2 === false) {
        gpuCompute.setDataType(THREE.HalfFloatType); // Fallback for WebGL1
    }

    // Create initial state textures
    const dtPosition = gpuCompute.createTexture();
    const dtVelocity = gpuCompute.createTexture();
    fillPositionTexture(dtPosition);
    fillVelocityTexture(dtVelocity);

    // Add texture variables
    velocityVariable = gpuCompute.addVariable("textureVelocity", velocityShader, dtVelocity);
    positionVariable = gpuCompute.addVariable("texturePosition", positionShader, dtPosition);

    // Add dependencies and uniforms
    gpuCompute.setVariableDependencies(velocityVariable, [positionVariable, velocityVariable]);
    gpuCompute.setVariableDependencies(positionVariable, [positionVariable, velocityVariable]);

    positionUniforms = positionVariable.material.uniforms;
    velocityUniforms = velocityVariable.material.uniforms;

    velocityUniforms["time"] = { value: 0.0 };
    velocityUniforms["delta"] = { value: 0.0 }; // Delta time uniform
    velocityUniforms["separationDistance"] = { value: PARAMS.separationDistance };
    velocityUniforms["alignmentDistance"] = { value: PARAMS.alignmentDistance };
    velocityUniforms["cohesionDistance"] = { value: PARAMS.cohesionDistance };
    velocityUniforms["freedomFactor"] = { value: PARAMS.freedomFactor };

    velocityUniforms["foodPositions"] = { value: new Array(MAX_FOOD).fill(new THREE.Vector3()) };
    velocityUniforms["foodActive"] = { value: new Array(MAX_FOOD).fill(false) };
    velocityUniforms["predatorPositions"] = { value: new Array(MAX_PREDATORS).fill(new THREE.Vector3()) };
    velocityUniforms["predatorActive"] = { value: new Array(MAX_PREDATORS).fill(false) };
    velocityUniforms["foodWeight"] = { value: PARAMS.foodWeight };
    velocityUniforms["predatorWeight"] = { value: PARAMS.predatorWeight };
    velocityUniforms["foodRadius"] = { value: PARAMS.foodRadius };
    velocityUniforms["foodOrbitSpeed"] = { value: PARAMS.foodOrbitSpeed };
    velocityUniforms["predatorRadius"] = { value: PARAMS.predatorRadius };

    velocityUniforms["separationWeight"] = { value: PARAMS.separationWeight };
    velocityUniforms["alignmentWeight"] = { value: PARAMS.alignmentWeight };
    velocityUniforms["cohesionWeight"] = { value: PARAMS.cohesionWeight };
    velocityUniforms["predatorWeight"] = { value: PARAMS.predatorWeight };
    velocityUniforms["maxSpeed"] = { value: PARAMS.maxSpeed };
    velocityUniforms["maxSteerForce"] = { value: PARAMS.maxSteerForce };

    positionUniforms["delta"] = { value: 0.0 }; // Delta time uniform


    // Check for completeness
    const error = gpuCompute.init();
    if (error !== null) {
        console.error('GPUComputationRenderer Error: ' + error);
    }
}

function fillPositionTexture(texture) {
    const theArray = texture.image.data;
    for (let k = 0, kl = theArray.length; k < kl; k += 4) {
        const x = Math.random() * BOUNDS - BOUNDS_HALF;
        const y = Math.random() * BOUNDS - BOUNDS_HALF;
        const z = Math.random() * BOUNDS - BOUNDS_HALF;
        theArray[k + 0] = x;
        theArray[k + 1] = y;
        theArray[k + 2] = z;
        theArray[k + 3] = 1; // W component (often unused but required)
    }
}

function fillVelocityTexture(texture) {
    const theArray = texture.image.data;
    for (let k = 0, kl = theArray.length; k < kl; k += 4) {
        const x = Math.random() - 0.5;
        const y = Math.random() - 0.5;
        const z = Math.random() - 0.5;
        theArray[k + 0] = x * 10;
        theArray[k + 1] = y * 10;
        theArray[k + 2] = z * 10;
        theArray[k + 3] = 1;
    }
}

function initBirds() {
    const birdMaterial = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.6,
        metalness: 0.2,
        // emissive: 0x333333, // Subtle glow
        side: THREE.DoubleSide
    });

    birdMesh = new THREE.InstancedMesh(BirdGeometry, birdMaterial, NUM_BIRDS);
    birdMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); // Important for performance
    scene.add(birdMesh);

    // Set initial scale for all instances
    const dummy = new THREE.Object3D();
    dummy.scale.set(PARAMS.birdScale, PARAMS.birdScale, PARAMS.birdScale);
    dummy.updateMatrix();
    for (let i = 0; i < NUM_BIRDS; i++) {
        birdMesh.setMatrixAt(i, dummy.matrix);
    }
    birdMesh.instanceMatrix.needsUpdate = true;
}


function initLighting() {
    const ambientLight = new THREE.AmbientLight(0x404060, 1.0); // Soft ambient light
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffccaa, 3.0); // Warm sunlight
    sunLight.position.set(0.5, 1, 0.75); // Angled sun
    sunLight.castShadow = false; // Shadows are expensive with many objects
    scene.add(sunLight);

    // Optional: Add a subtle backlight for rim lighting effect
    // const rimLight = new THREE.DirectionalLight(0x6080ff, 0.5);
    // rimLight.position.set(-0.5, -0.5, -1);
    // scene.add(rimLight);
}

function initEnvironment() {
    // Simple Ground Plane (optional)
    // const groundGeo = new THREE.PlaneGeometry(BOUNDS * 1.5, BOUNDS * 1.5);
    // const groundMat = new THREE.MeshStandardMaterial({ color: 0x101015, roughness: 0.9 });
    // const ground = new THREE.Mesh(groundGeo, groundMat);
    // ground.rotation.x = -Math.PI / 2;
    // ground.position.y = -BOUNDS_HALF * 0.8; // Slightly below the simulation center
    // scene.add(ground);

    // Skybox
    // const loader = new THREE.CubeTextureLoader();
    // const texture = loader.load([
    //     'https://threejs.org/examples/textures/cube/Bridge2/posx.jpg', 'https://threejs.org/examples/textures/cube/Bridge2/negx.jpg',
    //     'https://threejs.org/examples/textures/cube/Bridge2/posy.jpg', 'https://threejs.org/examples/textures/cube/Bridge2/negy.jpg',
    //     'https://threejs.org/examples/textures/cube/Bridge2/posz.jpg', 'https://threejs.org/examples/textures/cube/Bridge2/negz.jpg',
    //     // Replace with a more atmospheric skybox (e.g., sunset/dusk) for better mood
    //     // Example: 'px.png', 'nx.png', 'py.png', 'ny.png', 'pz.png', 'nz.png'
    // ]);
    // scene.background = texture;
    // scene.environment = texture; // For reflections on birds

    // --- SET Background Color ---
    scene.background = new THREE.Color(0x000005); // Very dark blue/black
    scene.environment = null; // Or set to a minimal environment map if needed for bird reflections
    
    // Simple Obstacle (Example)
     const obstacleGeo = new THREE.SphereGeometry(BOUNDS * 0.1, 32, 32);
     const obstacleMat = new THREE.MeshStandardMaterial({ color: 0x555566, roughness: 0.8 });
     const obstacle = new THREE.Mesh(obstacleGeo, obstacleMat);
     obstacle.position.set(0, 0, 0); // Center obstacle
     // scene.add(obstacle); // Uncomment to add obstacle (avoidance logic not fully implemented in shader yet)
}

// --- Create Starfield ---
function initStarfield() {
    const starCount = 15000; // Number of stars
    const starSphereRadius = BOUNDS * 5; // Make stars very distant

    const positions = [];
    const colors = [];
    const starGeometry = new THREE.BufferGeometry();
    const color = new THREE.Color();

    // Create a simple white radial gradient texture for stars
    function createStarTexture() {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        const size = 64;
        canvas.width = size;
        canvas.height = size;

        const gradient = context.createRadialGradient(
            size / 2, size / 2, 0, // Inner circle (center, radius 0)
            size / 2, size / 2, size / 2 // Outer circle (center, radius size/2)
        );
        gradient.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)'); // White center
        gradient.addColorStop(0.3, 'rgba(220, 220, 255, 0.8)'); // Faint blueish glow
        gradient.addColorStop(1.0, 'rgba(0, 0, 0, 0)');       // Transparent edge

        context.fillStyle = gradient;
        context.fillRect(0, 0, size, size);

        return new THREE.CanvasTexture(canvas);
    }

    const starTexture = createStarTexture();

    for (let i = 0; i < starCount; i++) {
        // Generate random spherical coordinates
        const phi = Math.acos(-1 + (2 * Math.random())); // Latitude distribution
        const theta = Math.random() * 2 * Math.PI;       // Longitude

        // Convert to Cartesian coordinates
        const x = starSphereRadius * Math.sin(phi) * Math.cos(theta);
        const y = starSphereRadius * Math.sin(phi) * Math.sin(theta);
        const z = starSphereRadius * Math.cos(phi);

        positions.push(x, y, z);

        // Add subtle color variation
        const variability = Math.random();
        if (variability < 0.1) {
            color.setHSL(0.6, 0.9, Math.random() * 0.2 + 0.8); // Pale Blue
        } else if (variability < 0.2) {
            color.setHSL(0.1, 0.9, Math.random() * 0.2 + 0.75); // Pale Yellow/Orange
        } else {
            color.setHSL(0, 0, Math.random() * 0.3 + 0.7); // Mostly White/Greyish
        }
        colors.push(color.r, color.g, color.b);
    }

    starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    starGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const starMaterial = new THREE.PointsMaterial({
        size: 15,                    // Adjust base size of stars
        map: starTexture,            // Use the generated texture
        vertexColors: true,          // Use the colors attribute
        blending: THREE.AdditiveBlending, // Crucial for glow effect
        depthWrite: false,           // Prevents stars occluding each other oddly
        transparent: true,           // Needed for blending and texture alpha
        sizeAttenuation: true        // Stars further away appear smaller (optional, could be false)
    });

    const starField = new THREE.Points(starGeometry, starMaterial);
    scene.add(starField);

    console.log(`Initialized ${starCount} stars.`);
}

function initPostProcessing() {
    composer = new EffectComposer(renderer);

    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);

    bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        PARAMS.bloomStrength, // strength
        PARAMS.bloomRadius,  // radius
        PARAMS.bloomThreshold // threshold
    );
    composer.addPass(bloomPass);

    // FXAA for anti-aliasing (lighter than SMAA or TAA)
    fxaaPass = new ShaderPass(FXAAShader);
    const pixelRatio = renderer.getPixelRatio();
    fxaaPass.material.uniforms['resolution'].value.x = 1 / (window.innerWidth * pixelRatio);
    fxaaPass.material.uniforms['resolution'].value.y = 1 / (window.innerHeight * pixelRatio);
    composer.addPass(fxaaPass);
}

// --- Helper Function to get random position within bounds ---
function getRandomPosition() {
    return new THREE.Vector3(
        Math.random() * BOUNDS - BOUNDS_HALF,
        Math.random() * BOUNDS * 0.6 - BOUNDS_HALF * 0.3, // Keep them slightly lower/higher? Adjust Y range
        Math.random() * BOUNDS - BOUNDS_HALF
    );
}

// --- Initialize Attractors and Predators ---
function initAttractorsPredators() {
    // Food Sources
    const foodGeo = new THREE.SphereGeometry(6, 16, 8); // Slightly larger than predator visual
    const foodMat = new THREE.MeshBasicMaterial({ color: 0x40ff40 }); // Green

    for (let i = 0; i < MAX_FOOD; i++) {
        const orbitParams = {
            radius: BOUNDS_HALF * (0.3 + Math.random() * 0.6), // Orbit radius (30% to 90% of bounds)
            speedFactor: 0.8 + Math.random() * 0.4, // Individual speed variation (80% to 120% of base)
            phase: Math.random() * Math.PI * 2,     // Starting angle
            vertAmp: BOUNDS_HALF * (0.1 + Math.random() * 0.3), // Vertical movement amplitude
            vertFreq: 0.3 + Math.random() * 0.4,   // Vertical oscillation speed (slower than orbit)
            vertPhase: Math.random() * Math.PI * 2 // Starting vertical phase
        };

        // Calculate initial position (using time = 0 for simplicity here)
        const initialPos = new THREE.Vector3(
            orbitParams.radius * Math.cos(orbitParams.phase),
            orbitParams.vertAmp * Math.sin(orbitParams.vertPhase), // Use vertPhase for initial Y
            orbitParams.radius * Math.sin(orbitParams.phase)
        );

        const foodMesh = new THREE.Mesh(foodGeo, foodMat);
        foodMesh.position.copy(initialPos);
        // scene.add(foodMesh);

        foodSources.push({
            position: initialPos.clone(), // Store initial position
            orbitParams: orbitParams,     // Store the generated orbit parameters
            active: true,
            visual: foodMesh
            // Removed: startPosition, targetPosition, timer
        });

        // Initialize uniforms immediately with the starting position
        velocityUniforms.foodPositions.value[i] = initialPos.clone(); // CLONE!
        velocityUniforms.foodActive.value[i] = true;
    }

    // Predator Sources
    const predGeo = new THREE.SphereGeometry(5, 16, 8);
    const predMat = new THREE.MeshBasicMaterial({ color: 0xff4040 }); // Red

    for (let i = 0; i < MAX_PREDATORS; i++) {
        const position = getRandomPosition();
        const predMesh = new THREE.Mesh(predGeo, predMat);
        predMesh.position.copy(position);
        // scene.add(predMesh);

        predatorSources.push({
            position: position,
            timer: Math.random() * PREDATOR_LIFETIME, // Start with random timers
            active: true,
            visual: predMesh
        });
        // Initialize uniforms immediately
        velocityUniforms.predatorPositions.value[i] = position.clone(); // CLONE!
        velocityUniforms.predatorActive.value[i] = true;
    }
}


// --- Update Attractors and Predators (Call this in animate loop) ---
function updateAttractorsPredators(delta, time) {
    // Update Food
    for (let i = 0; i < MAX_FOOD; i++) {
        const food = foodSources[i];
        if (food.active) {
            const params = food.orbitParams;

            // Calculate current angle based on global time, base speed, individual speed factor, and phase
            const angle = time * PARAMS.foodOrbitSpeed * params.speedFactor + params.phase;

            // Calculate XZ position (orbiting around Y axis)
            const x = params.radius * Math.cos(angle);
            const z = params.radius * Math.sin(angle);

            // Calculate Y position (vertical oscillation)
            const y = params.vertAmp * Math.sin(time * params.vertFreq * PARAMS.foodOrbitSpeed + params.vertPhase);

            // Update the food source's logical position
            food.position.set(x, y, z);

            // Update the visual mesh position
            food.visual.position.copy(food.position);

            // --- IMPORTANT: Update the uniform for the GPU ---
            velocityUniforms.foodPositions.value[i].copy(food.position);
        }
        // Could add logic here to deactivate/reactivate food if needed
    }

    // Update Predators
    for (let i = 0; i < MAX_PREDATORS; i++) {
        const pred = predatorSources[i];
        if (pred.active) {
            pred.timer -= delta;
            if (pred.timer <= 0) {
                // Relocate predator
                pred.position.copy(getRandomPosition());
                pred.timer = PREDATOR_LIFETIME + Math.random() * 8.0 - 4.0; // Reset timer
                pred.visual.position.copy(pred.position);

                // Update uniform
                velocityUniforms.predatorPositions.value[i].copy(pred.position);
            }
        }
    }
}

function initGUI() {
    gui = new GUI();

    const boidFolder = gui.addFolder('Boid Behavior');
    boidFolder.add(PARAMS, 'separationDistance', 1, 100, 0.1).onChange(updateUniforms);
    boidFolder.add(PARAMS, 'alignmentDistance', 1, 100, 0.1).onChange(updateUniforms);
    boidFolder.add(PARAMS, 'cohesionDistance', 1, 100, 0.1).onChange(updateUniforms);
    boidFolder.add(PARAMS, 'freedomFactor', 0, 2, 0.01).onChange(updateUniforms);
    boidFolder.add(PARAMS, 'separationWeight', 0, 5, 0.1).onChange(updateUniforms);
    boidFolder.add(PARAMS, 'alignmentWeight', 0, 5, 0.1).onChange(updateUniforms);
    boidFolder.add(PARAMS, 'cohesionWeight', 0, 5, 0.1).onChange(updateUniforms);
    boidFolder.add(PARAMS, 'foodWeight', 0, 10, 0.1).onChange(updateUniforms);
    boidFolder.add(PARAMS, 'predatorWeight', 0, 15, 0.1).name('Auto Predator Weight').onChange(updateUniforms); // Add new one or rename
    boidFolder.add(PARAMS, 'foodRadius', 10, 800, 1).onChange(updateUniforms);
    boidFolder.add(PARAMS, 'predatorRadius', 10, 200, 1).onChange(updateUniforms);
    boidFolder.add(PARAMS, 'maxSpeed', 1, 200, 1).onChange(updateUniforms);
    boidFolder.add(PARAMS, 'maxSteerForce', 0.1, 40, 0.1).onChange(updateUniforms); // Smaller step for finer control
    boidFolder.add(PARAMS, 'foodOrbitSpeed', 0.01, 5.0, 0.001).onChange(updateUniforms);
    boidFolder.close(); // Start closed

    const visualFolder = gui.addFolder('Visuals');
    visualFolder.add(PARAMS, 'birdScale', 0.1, 3.0, 0.05).onChange(v => {
        // Update scale requires recreating the dummy matrix and applying to all
        const dummy = new THREE.Object3D();
        dummy.scale.set(v, v, v);
        dummy.updateMatrix();
        for (let i = 0; i < NUM_BIRDS; i++) {
            birdMesh.setMatrixAt(i, dummy.matrix); // Only updates scale part implicitly if rotation/pos is updated later
        }
        birdMesh.instanceMatrix.needsUpdate = true; // Might need full update in render loop
    });
    visualFolder.add(PARAMS, 'bloomThreshold', 0, 1, 0.01).onChange(v => bloomPass.threshold = v);
    visualFolder.add(PARAMS, 'bloomStrength', 0, 3, 0.01).onChange(v => bloomPass.strength = v);
    visualFolder.add(PARAMS, 'bloomRadius', 0, 1, 0.01).onChange(v => bloomPass.radius = v);
    visualFolder.add(scene.fog, 'density', 0, 0.01, 0.0001).name('Fog Density');
    visualFolder.close();

    const simulationFolder = gui.addFolder('Simulation');
    simulationFolder.add(PARAMS, 'timeScale', 0.0, 10.0, 0.01).name('Time Scale');

    gui.close(); // Start GUI closed
}

function updateUniforms() {
    velocityUniforms["separationDistance"].value = PARAMS.separationDistance;
    velocityUniforms["alignmentDistance"].value = PARAMS.alignmentDistance;
    velocityUniforms["cohesionDistance"].value = PARAMS.cohesionDistance;
    velocityUniforms["freedomFactor"].value = PARAMS.freedomFactor;
    velocityUniforms["separationWeight"].value = PARAMS.separationWeight;
    velocityUniforms["alignmentWeight"].value = PARAMS.alignmentWeight;
    velocityUniforms["cohesionWeight"].value = PARAMS.cohesionWeight;
    velocityUniforms["predatorWeight"].value = PARAMS.predatorWeight;
    velocityUniforms["maxSpeed"].value = PARAMS.maxSpeed;
    velocityUniforms["maxSteerForce"].value = PARAMS.maxSteerForce;

    // ADD/UPDATE New Uniforms
    velocityUniforms["foodWeight"].value = PARAMS.foodWeight;
    velocityUniforms["predatorWeight"].value = PARAMS.predatorWeight;
    velocityUniforms["foodRadius"].value = PARAMS.foodRadius;
    velocityUniforms["foodOrbitSpeed"].value = PARAMS.foodOrbitSpeed;
    velocityUniforms["predatorRadius"].value = PARAMS.predatorRadius;
}


// --- Event Handlers ---

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);

    const pixelRatio = renderer.getPixelRatio();
    fxaaPass.material.uniforms['resolution'].value.x = 1 / (window.innerWidth * pixelRatio);
    fxaaPass.material.uniforms['resolution'].value.y = 1 / (window.innerHeight * pixelRatio);
}


function onPointerMove(event) {
    // Update mouse normalized coordinates
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    // If predator is active (mouse button down), update its position
    if (PARAMS.hasPredator) {
        updatePredatorPosition();
    }
}


function onPointerDown(event) {
     event.preventDefault();

    // Use raycaster to find intersection point on the invisible plane
    updatePredatorPosition();

    // If intersection occurred, activate predator
     if (PARAMS.predator.length() > 0) { // Check if intersection was valid
        velocityUniforms["hasPredator"].value = true;
        PARAMS.hasPredator = true;
     } else {
         velocityUniforms["hasPredator"].value = false;
         PARAMS.hasPredator = false;
     }
}

function onPointerUp(event) {
    event.preventDefault();
    velocityUniforms["hasPredator"].value = false;
    PARAMS.hasPredator = false;
}

function updatePredatorPosition() {
     raycaster.setFromCamera(mouse, camera);
     const intersects = raycaster.intersectObject(targetMesh);

     if (intersects.length > 0) {
         const point = intersects[0].point;
         PARAMS.predator.copy(point); // Store the 3D world position
         velocityUniforms["predator"].value.copy(PARAMS.predator);
     } else {
         // If no intersection (e.g., pointing outside the plane), maybe deactivate predator?
         // Or keep last known valid position? For now, let's invalidate:
         PARAMS.predator.set(0, -10000, 0); // Move it far away effectively disabling it
         velocityUniforms["predator"].value.copy(PARAMS.predator);
         // We might also explicitly set hasPredator to false here if pointer is down
         // velocityUniforms[ "hasPredator" ].value = false;
         // PARAMS.hasPredator = false;
     }
}


// --- Animation Loop ---

function animate() {
    requestAnimationFrame(animate);

    const delta = Math.min(clock.getDelta(), 0.1) * PARAMS.timeScale; // Get delta time, clamp max, apply time scale
    const time = clock.elapsedTime;

    // Update Controls
    controls.update();

    updateAttractorsPredators(delta, time);

    // Update GPU Compute Uniforms
    velocityUniforms["time"].value = time;
    velocityUniforms["delta"].value = delta;
    positionUniforms["delta"].value = delta;

    // Run GPU Compute
    gpuCompute.compute();

    // Update Instanced Mesh
    updateBirdInstances();

    // Render Scene with Post-Processing
    composer.render();

    // Optional: Update Stats
    // stats.update();
}

function updateBirdInstances() {
    const positionTexture = gpuCompute.getCurrentRenderTarget(positionVariable).texture;
    const velocityTexture = gpuCompute.getCurrentRenderTarget(velocityVariable).texture;

    // Note: Reading back texture data to the CPU is SLOW!
    // A more advanced approach uses vertex texture fetching in the bird's vertex shader
    // to directly read position/velocity from the textures on the GPU.
    // This CPU-based update is simpler to implement but limits performance.

    // Create a DataTexture to read GPU texture data (can be reused)
    // This is still suboptimal but better than readRenderTargetPixels
    // const positionArray = new Float32Array(NUM_BIRDS * 4);
    // renderer.readRenderTargetPixels(gpuCompute.getCurrentRenderTarget(positionVariable), 0, 0, WIDTH, WIDTH, positionArray);
    // const velocityArray = new Float32Array(NUM_BIRDS * 4);
    // renderer.readRenderTargetPixels(gpuCompute.getCurrentRenderTarget(velocityVariable), 0, 0, WIDTH, WIDTH, velocityArray);

    // *** OPTIMIZATION POINT ***
    // The code below reads textures back to CPU, which is slow.
    // For a *truly* high-performance demo, you'd pass the position/velocity
    // textures directly to a custom ShaderMaterial on the InstancedMesh
    // and calculate the instance matrix in the vertex shader using gl_InstanceID
    // and texture2D lookups.

    // Simplified CPU update (demonstration purposes):
    const dummy = new THREE.Object3D();
    // const positionArray = gpuCompute.getCurrentRenderTarget(positionVariable).texture.image.data; // <-- REMOVE THIS LINE
    // const velocityArray = gpuCompute.getCurrentRenderTarget(velocityVariable).texture.image.data; // <-- REMOVE THIS LINE

    // Correct (but slow) CPU readback:
    const positionBuffer = new Float32Array(NUM_BIRDS * 4); // Allocate CPU buffer
    renderer.readRenderTargetPixels(
        gpuCompute.getCurrentRenderTarget(positionVariable), // Source RenderTarget
        0, 0, WIDTH, WIDTH,                                 // Area to read (x, y, width, height)
        positionBuffer                                      // Destination buffer
    );

    const velocityBuffer = new Float32Array(NUM_BIRDS * 4); // Allocate CPU buffer
    renderer.readRenderTargetPixels(
        gpuCompute.getCurrentRenderTarget(velocityVariable), // Source RenderTarget
        0, 0, WIDTH, WIDTH,                                 // Area to read (x, y, width, height)
        velocityBuffer                                      // Destination buffer
    );


    for (let i = 0; i < NUM_BIRDS; i++) {
        const index = i * 4;

        // Get position and velocity from texture data buffers
        dummy.position.set(positionBuffer[index], positionBuffer[index + 1], positionBuffer[index + 2]); // Use positionBuffer
        const velocity = new THREE.Vector3(velocityBuffer[index], velocityBuffer[index + 1], velocityBuffer[index + 2]); // Use velocityBuffer

        // Orientation: Point the bird in the direction of velocity
        if (velocity.lengthSq() > 0.001) { // Avoid issues with zero velocity
            dummy.lookAt(dummy.position.clone().add(velocity));
        }

        // Apply fixed scale
        dummy.scale.setScalar(PARAMS.birdScale);

        // Update the instance matrix
        dummy.updateMatrix();
        birdMesh.setMatrixAt(i, dummy.matrix);
    }

    birdMesh.instanceMatrix.needsUpdate = true;
}


// --- Start ---
init();
animate();