import React, { useMemo, useRef, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * Ambient 3D scene: a slowly-breathing wireframe "research core"
 * surrounded by a drifting particle constellation, with gentle
 * mouse parallax. Subtle by design — low opacity, capped DPR,
 * and paused rendering cost via frameloop="always" but tiny geometry.
 */

const PARTICLE_COUNT = 900;

function ParticleField() {
  const ref = useRef<THREE.Points>(null);

  const [positions, sizes] = useMemo(() => {
    const pos = new Float32Array(PARTICLE_COUNT * 3);
    const size = new Float32Array(PARTICLE_COUNT);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Distribute in a flattened ellipsoid shell around the core
      const r = 2.2 + Math.random() * 4.5;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta) * 1.6;
      pos[i * 3 + 1] = r * Math.cos(phi) * 0.7;
      pos[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
      size[i] = Math.random();
    }
    return [pos, size];
  }, []);

  useFrame((state) => {
    if (!ref.current) return;
    const t = state.clock.elapsedTime;
    ref.current.rotation.y = t * 0.02;
    ref.current.rotation.x = Math.sin(t * 0.05) * 0.06;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-size" args={[sizes, 1]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.035}
        color="#cc785c"
        transparent
        opacity={0.55}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

function ResearchCore() {
  const outer = useRef<THREE.Mesh>(null);
  const inner = useRef<THREE.Mesh>(null);
  const halo = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (outer.current) {
      outer.current.rotation.y = t * 0.08;
      outer.current.rotation.z = t * 0.03;
      const breathe = 1 + Math.sin(t * 0.6) * 0.035;
      outer.current.scale.setScalar(breathe);
    }
    if (inner.current) {
      inner.current.rotation.y = -t * 0.14;
      inner.current.rotation.x = t * 0.06;
    }
    if (halo.current) {
      halo.current.rotation.z = t * 0.05;
    }
  });

  return (
    <group>
      <mesh ref={outer}>
        <icosahedronGeometry args={[1.55, 1]} />
        <meshBasicMaterial
          color="#cc785c"
          wireframe
          transparent
          opacity={0.28}
        />
      </mesh>
      <mesh ref={inner}>
        <icosahedronGeometry args={[0.95, 0]} />
        <meshBasicMaterial
          color="#e8b288"
          wireframe
          transparent
          opacity={0.35}
        />
      </mesh>
      <mesh ref={halo} rotation={[Math.PI / 2.4, 0, 0]}>
        <torusGeometry args={[2.6, 0.006, 8, 96]} />
        <meshBasicMaterial color="#d4a276" transparent opacity={0.4} />
      </mesh>
      <mesh rotation={[Math.PI / 1.8, 0.4, 0]}>
        <torusGeometry args={[3.1, 0.004, 8, 96]} />
        <meshBasicMaterial color="#4a9e99" transparent opacity={0.22} />
      </mesh>
    </group>
  );
}

function ParallaxRig() {
  const { camera, pointer } = useThree();
  useFrame(() => {
    // Ease the camera toward a gentle offset of the pointer position
    camera.position.x += (pointer.x * 0.55 - camera.position.x) * 0.04;
    camera.position.y += (pointer.y * 0.35 - camera.position.y) * 0.04;
    camera.lookAt(0, 0, 0);
  });
  return null;
}

export const HeroScene: React.FC = () => {
  const reduceMotion =
    typeof window !== 'undefined' &&
    window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

  if (reduceMotion) return null;

  return (
    <div className="hero-3d-canvas" aria-hidden="true">
      <Canvas
        camera={{ position: [0, 0, 7], fov: 46 }}
        dpr={[1, 1.5]}
        gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
      >
        <Suspense fallback={null}>
          <ParticleField />
          <ResearchCore />
          <ParallaxRig />
        </Suspense>
      </Canvas>
    </div>
  );
};
