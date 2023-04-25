import { OrbitControls, useGLTF } from '@react-three/drei';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { gsap } from 'gsap';
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import './App.css';
let diskArray;

const Un = {
  vertexShader:
    '\nvarying vec2 vUv; \nvarying vec3 vNormal;\nvarying vec3 vPosition;\nvarying vec2 vN;\n\nuniform float u_time;\nuniform float u_progess;\nuniform float pi;\nuniform int u_dispFunc;\n\nuniform float u_f1speed;\nuniform float u_f1freq;\nuniform float u_f1amp;\n\nuniform float u_f2speed;\nuniform float u_f2freqy;\nuniform float u_f2freqrad;\nuniform float u_f2ampy;\nuniform float u_f2amprad;\n\nuniform float u_f3speed;\nuniform float u_f3freqy;\nuniform float u_f3freqrad;\nuniform float u_f3ampy;\nuniform float u_f3amprad;\n\nuniform float u_f4speed;\nuniform float u_f4freq;\nuniform float u_f4amp;\n\nuniform vec3 u_bumpPos;\nuniform float u_bumpInt;\nuniform float u_bumRadius;\nuniform float u_bumAmp;\n\nfloat tangentFactor = 0.005;\n\n\n'.concat(
      {
        noise4:
          '\n  vec4 mod289(vec4 x) {\n    return x - floor(x * (1.0 / 289.0)) * 289.0; }\n  \n  float mod289(float x) {\n    return x - floor(x * (1.0 / 289.0)) * 289.0; }\n  \n  vec4 permute(vec4 x) {\n       return mod289(((x*34.0)+1.0)*x);\n  }\n  \n  float permute(float x) {\n       return mod289(((x*34.0)+1.0)*x);\n  }\n  \n  vec4 taylorInvSqrt(vec4 r)\n  {\n    return 1.79284291400159 - 0.85373472095314 * r;\n  }\n  \n  float taylorInvSqrt(float r)\n  {\n    return 1.79284291400159 - 0.85373472095314 * r;\n  }\n  \n  vec4 grad4(float j, vec4 ip)\n    {\n    const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);\n    vec4 p,s;\n  \n    p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;\n    p.w = 1.5 - dot(abs(p.xyz), ones.xyz);\n    s = vec4(lessThan(p, vec4(0.0)));\n    p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;\n  \n    return p;\n    }\n  \n  // (sqrt(5) - 1)/4 = F4, used once below\n  #define F4 0.309016994374947451\n  \n  float snoise(vec4 v)\n    {\n    const vec4  C = vec4( 0.138196601125011,  // (5 - sqrt(5))/20  G4\n                          0.276393202250021,  // 2 * G4\n                          0.414589803375032,  // 3 * G4\n                         -0.447213595499958); // -1 + 4 * G4\n  \n  // First corner\n    vec4 i  = floor(v + dot(v, vec4(F4)) );\n    vec4 x0 = v -   i + dot(i, C.xxxx);\n  \n  // Other corners\n  \n  // Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)\n    vec4 i0;\n    vec3 isX = step( x0.yzw, x0.xxx );\n    vec3 isYZ = step( x0.zww, x0.yyz );\n  //  i0.x = dot( isX, vec3( 1.0 ) );\n    i0.x = isX.x + isX.y + isX.z;\n    i0.yzw = 1.0 - isX;\n  //  i0.y += dot( isYZ.xy, vec2( 1.0 ) );\n    i0.y += isYZ.x + isYZ.y;\n    i0.zw += 1.0 - isYZ.xy;\n    i0.z += isYZ.z;\n    i0.w += 1.0 - isYZ.z;\n  \n    // i0 now contains the unique values 0,1,2,3 in each channel\n    vec4 i3 = clamp( i0, 0.0, 1.0 );\n    vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );\n    vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );\n  \n    //  x0 = x0 - 0.0 + 0.0 * C.xxxx\n    //  x1 = x0 - i1  + 1.0 * C.xxxx\n    //  x2 = x0 - i2  + 2.0 * C.xxxx\n    //  x3 = x0 - i3  + 3.0 * C.xxxx\n    //  x4 = x0 - 1.0 + 4.0 * C.xxxx\n    vec4 x1 = x0 - i1 + C.xxxx;\n    vec4 x2 = x0 - i2 + C.yyyy;\n    vec4 x3 = x0 - i3 + C.zzzz;\n    vec4 x4 = x0 + C.wwww;\n  \n  // Permutations\n    i = mod289(i);\n    float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);\n    vec4 j1 = permute( permute( permute( permute (\n               i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))\n             + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))\n             + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))\n             + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));\n  \n  // Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope\n  // 7*7*6 = 294, which is close to the ring size 17*17 = 289.\n    vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;\n  \n    vec4 p0 = grad4(j0,   ip);\n    vec4 p1 = grad4(j1.x, ip);\n    vec4 p2 = grad4(j1.y, ip);\n    vec4 p3 = grad4(j1.z, ip);\n    vec4 p4 = grad4(j1.w, ip);\n  \n  // Normalise gradients\n    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));\n    p0 *= norm.x;\n    p1 *= norm.y;\n    p2 *= norm.z;\n    p3 *= norm.w;\n    p4 *= taylorInvSqrt(dot(p4,p4));\n  \n  // Mix contributions from the five corners\n    vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);\n    vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);\n    m0 = m0 * m0;\n    m1 = m1 * m1;\n    return 49.0 * ( dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))\n                 + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;\n  \n    }\n',
        noise3:
          '\nvec3 mod289(vec3 x) {\n  return x - floor(x * (1.0 / 289.0)) * 289.0;\n}\n\nvec4 mod289(vec4 x) {\n  return x - floor(x * (1.0 / 289.0)) * 289.0;\n}\n\nvec4 permute(vec4 x) {\n     return mod289(((x*34.0)+1.0)*x);\n}\n\nvec4 taylorInvSqrt(vec4 r)\n{\n  return 1.79284291400159 - 0.85373472095314 * r;\n}\n\nfloat snoise(vec3 v)\n  {\n  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;\n  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);\n\n// First corner\n  vec3 i  = floor(v + dot(v, C.yyy) );\n  vec3 x0 =   v - i + dot(i, C.xxx) ;\n\n// Other corners\n  vec3 g = step(x0.yzx, x0.xyz);\n  vec3 l = 1.0 - g;\n  vec3 i1 = min( g.xyz, l.zxy );\n  vec3 i2 = max( g.xyz, l.zxy );\n\n  //   x0 = x0 - 0.0 + 0.0 * C.xxx;\n  //   x1 = x0 - i1  + 1.0 * C.xxx;\n  //   x2 = x0 - i2  + 2.0 * C.xxx;\n  //   x3 = x0 - 1.0 + 3.0 * C.xxx;\n  vec3 x1 = x0 - i1 + C.xxx;\n  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y\n  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y\n\n// Permutations\n  i = mod289(i);\n  vec4 p = permute( permute( permute(\n             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))\n           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))\n           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));\n\n// Gradients: 7x7 points over a square, mapped onto an octahedron.\n// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)\n  float n_ = 0.142857142857; // 1.0/7.0\n  vec3  ns = n_ * D.wyz - D.xzx;\n\n  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)\n\n  vec4 x_ = floor(j * ns.z);\n  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)\n\n  vec4 x = x_ *ns.x + ns.yyyy;\n  vec4 y = y_ *ns.x + ns.yyyy;\n  vec4 h = 1.0 - abs(x) - abs(y);\n\n  vec4 b0 = vec4( x.xy, y.xy );\n  vec4 b1 = vec4( x.zw, y.zw );\n\n  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;\n  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;\n  vec4 s0 = floor(b0)*2.0 + 1.0;\n  vec4 s1 = floor(b1)*2.0 + 1.0;\n  vec4 sh = -step(h, vec4(0.0));\n\n  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;\n  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;\n\n  vec3 p0 = vec3(a0.xy,h.x);\n  vec3 p1 = vec3(a0.zw,h.y);\n  vec3 p2 = vec3(a1.xy,h.z);\n  vec3 p3 = vec3(a1.zw,h.w);\n\n//Normalise gradients\n  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));\n  p0 *= norm.x;\n  p1 *= norm.y;\n  p2 *= norm.z;\n  p3 *= norm.w;\n\n// Mix final noise value\n  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);\n  m = m * m;\n  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),\n                                dot(p2,x2), dot(p3,x3) ) );\n  }\n',
      }.noise4,
      '\n\nfloat remap(float value, float low1, float high1, float low2, float high2){\n  return low2 + (value - low1) * (high2 - low2) / (high1 - low1);\n}\n\nvec3 orthogonal(vec3 v) {\n  return normalize(abs(v.x) > abs(v.z) ? vec3(-v.y, v.x, 0.0)\n  : vec3(0.0, -v.z, v.y));\n}\n\nvec3 func2(vec3 p, float speed, float freqy, float freqrad, float ampy, float amprad){\n  vec3 f2out = p; \n  f2out.xz = p.xz*remap(sin(p.y*freqy+u_time*0.001*speed+p.x*2.), -1., 1., 1.-ampy, 1.+ampy);\n  float rad = atan(p.x, p.z);\n  rad = remap(rad, -pi, pi, 00.,freqrad);\n  rad = sin(rad);\n  rad = remap(rad, -1., 1., 0., amprad);\n\n\n  float disp = sin((p.y*freqy+rad));      \n  f2out.xz *= remap(disp, 0., 1., 1.-ampy, 1.+ampy);\n  return f2out;\n}\n\nvec3 bumpyFyFunc(vec3 p, vec3 pOut, vec3 bPos, float bRad, float bInt, float bAmp){\n  float dist = distance(bPos, p);\n  dist = smoothstep(0.,bRad, dist);\n  dist = clamp(dist, 0., 1.);\n  dist = 1.-dist;\n  dist *= bInt;\n\n  vec3 bump = p*remap(dist, 0.,1.,1., 1.+bAmp);\n  return mix(pOut, bump, dist);\n\n}\n\nvec3 distorted(vec3 p) {\n  vec3 frame0 = p;\n  \n  vec3 frame1 = p*(1.+snoise(vec4(p*u_f1freq, u_time*0.0001*u_f1speed))*u_f1amp);\n\n  vec3 frame2 = func2(p,u_f2speed, u_f2freqy, u_f2freqrad,u_f2ampy, u_f2amprad);\n  \n  vec3 frame3 = func2(p,u_f3speed, u_f3freqy, u_f3freqrad,u_f3ampy, u_f3amprad);\n\n  vec3 frame4 = p*(1.+snoise(vec4(p*u_f4freq, u_time*0.0001*u_f4speed))*u_f4amp);\n\n  vec3 outputP = mix(frame0, frame1, clamp(u_progess, 0.,1.));\n  outputP = mix(outputP, frame2, clamp(u_progess-1., 0.,1.));\n  outputP = mix(outputP, frame3, clamp(u_progess-2., 0.,1.));\n  outputP = mix(outputP, frame4, clamp(u_progess-3., 0.,1.));\n\n  outputP = bumpyFyFunc(p, outputP, u_bumpPos, u_bumRadius, u_bumpInt, u_bumAmp);\n\n  return outputP;\n}\n\nvoid main() {\n  vUv = uv; \n  vNormal = normal;\n  vPosition = position;\n\n  vec3 dispPos = distorted(position);\n  vec3 tangent1 = orthogonal(normal);\n  vec3 tangent2 = normalize(cross(normal, tangent1));\n  vec3 nearby1 = position + tangent1 * tangentFactor;\n  vec3 nearby2 = position + tangent2 * tangentFactor;\n  vec3 distorted1 = distorted(nearby1);\n  vec3 distorted2 = distorted(nearby2);\n\n  vNormal = normalize(cross(distorted1 - dispPos, distorted2 - dispPos));\n\n  vec4 p = vec4( position, 1. );\n\n  vec3 e = normalize( vec3( modelViewMatrix * p ) );\n  vec3 n = normalize( normalMatrix * vNormal );\n\n\n  vec3 r = reflect( e, n );\n  float m = 2. * sqrt(\n    pow( r.x, 2. ) +\n    pow( r.y, 2. ) +\n    pow( r.z + 1., 2. )\n  );\n  vN = r.xy / m + .5;\n\n  vec4 modelViewPosition = modelViewMatrix * vec4(dispPos, 1.0);\n  // vec4 modelViewPosition = modelViewMatrix * vec4(position, 1.0);\n  gl_Position = projectionMatrix * modelViewPosition; \n}\n  '
    ),
  fragmentShader:
    '\nuniform sampler2D u_matCap;\nuniform vec3 u_bumpPos;\nuniform float u_bumpInt;\nuniform float pi;\nuniform float u_op;\n\nvarying vec2 vN;\nvarying vec3 vNormal;\nvarying vec3 vPosition;\nvarying vec2 vUv; \n\n\nfloat remap(float value, float low1, float high1, float low2, float high2){\n  return low2 + (value - low1) * (high2 - low2) / (high1 - low1);\n}\n\n\nvoid main() {\n\n  vec4 matCapCol = texture2D(u_matCap, vN);\n  matCapCol.a = u_op;\n\n  vec4 outcol = vec4(0.);\n  outcol.a = 1.;\n\n  float rad = atan(vPosition.x, vPosition.z);\n  rad = remap(rad, -pi, pi, 00.,10.*pi);\n  rad = sin(rad);\n  rad = remap(rad, -1., 1., 0., 1.);\n  // outPoint.xz = outPoint.xz*remap(sin((outPoint.y+rad)*10.), -1., 1., 1.-u_amp2, 1.+u_amp2);\n\n\n  outcol.rgb = vec3(sin((vPosition.y+rad*0.5)*10.+vPosition.x*10.));\n  // outcol.rgb = vec3(rad);\n\n  // gl_FragColor = outcol;\n  // gl_FragColor = vec4(vN, 1.,1.);\n  gl_FragColor = vec4(matCapCol);\n}\n  ',
};
const Disc = (props) => {
  console.log(Un.vertexShader);
  const mesh = useRef();
  const shader = useRef();
  const { scene } = useThree();
  const { nodes } = useGLTF('models/disc.glb');
  const texture = new THREE.TextureLoader().load('textures/matCapArtBall3.png');
  const mat = new THREE.MeshMatcapMaterial({ matcap: texture });
  const animTl = gsap.timeline({ paused: !0 });

  function generateDiskArray(t) {
    diskArray = new THREE.Group();
    for (var e = 0; e < 6; e++) {
      const r = new THREE.Object3D();
      const i = t.clone();
      console.log(i);
      i.scale.set(0, 0, 0);
      var n = 2 + 0.7 * e;
      animTl.to(i.scale, { x: n, y: 1, z: n, duration: 0.3 }, 0.1 * (5 - e));
      i.position.y = 0.2 * -e;
      r.add(i);
      diskArray.add(r);
    }
    diskArray.rotation.x = Math.PI / 2;
    scene.add(diskArray);
  }

  useEffect(() => {
    generateDiskArray(mesh.current);
  }, []);
  const jn = {
    mousePos: {
      x: 0,
      y: 0,
    },
  };
  function update(t) {
    if (diskArray) {
      for (var e = 0; e < diskArray.children.length; ) {
        const r = diskArray.children[e];
        const i = 1 + (Math.sin(0.0015 * Date.now() + 0.5 * e) + 1) / 20;
        const n = 0.001 * (diskArray.children.length - (e + 1));
        r.position.x += 0.05 * (jn.mousePos.x * n - r.position.x);

        r.position.z += 0.05 * (jn.mousePos.y * n - r.position.z);

        r.scale.set(i, 1, i);
        e++;
      }
      animTl && animTl.time(Math.min(Math.max(t - 4.5, 0), animTl.duration()));
    }
  }

  const onPointerMove = (e) => {
    jn.mousePos.x = e.clientX;
    jn.mousePos.y = e.clientY;
    update(1);
  };
  const $n = {
    frame1: { speed: 1.04, freq: 0.3, amp: 0.55 },
    frame2: { speed: 1.04, freqy: 5.19, freqrad: 8.5, ampy: 0.03, amprad: 5.78 },
    frame3: { speed: 3.9, freqy: 6.29, freqrad: 12.7, ampy: 0.02, amprad: 10 },
    frame4: { speed: 1.04, freq: 0.3, amp: 0.55 },
  };
  const uniform = useMemo(
    () => ({
      u_matCap: { value: texture },
      u_time: { value: 0 },
      pi: { value: Math.PI },
      u_progess: { value: 0 },
      u_op: { value: 1 },
      u_f1speed: { value: $n.frame1.speed },
      u_f1freq: { value: $n.frame1.freq },
      u_f1amp: { value: $n.frame1.amp },
      u_f2speed: { value: $n.frame2.speed },
      u_f2freqy: { value: $n.frame2.freqy },
      u_f2freqrad: { value: $n.frame2.freqrad },
      u_f2ampy: { value: $n.frame2.ampy },
      u_f2amprad: { value: $n.frame2.amprad },
      u_f3speed: { value: $n.frame3.speed },
      u_f3freqy: { value: $n.frame3.freqy },
      u_f3freqrad: { value: $n.frame3.freqrad },
      u_f3ampy: { value: $n.frame3.ampy },
      u_f3amprad: { value: $n.frame3.amprad },
      u_f4speed: { value: $n.frame4.speed },
      u_f4freq: { value: $n.frame4.freq },
      u_f4amp: { value: $n.frame4.amp },
      u_dispFunc: { value: 2 },
      u_bumpPos: { value: new THREE.Vector3() },
      u_bumpInt: { value: 0 },
      u_bumRadius: { value: 3 },
      u_bumAmp: { value: 0.4 },
    }),
    []
  );

  useFrame((state, delta) => {
    if (shader.current) {
      shader.current.uniforms.u_time.value += delta;
      shader.current.uniforms.u_bumpPos.value.x += 0.05 * (this.pointInt.x - shader.current.uniforms.u_bumpPos.value.x);
      shader.current.uniforms.u_bumpPos.value.y += 0.05 * (this.pointInt.y - shader.current.uniforms.u_bumpPos.value.y);
      shader.current.uniforms.u_bumpPos.value.z += 0.05 * (this.pointInt.z - shader.current.uniforms.u_bumpPos.value.z);
      shader.current.uniforms.u_bumpInt.value += 0.05 * (this.mIntTarget - shader.current.uniforms.u_bumpInt.value);
      shader.current.uniforms.u_progess.value = delta;
    }
  });

  return (
    <mesh onPointerMove={onPointerMove} ref={mesh} geometry={nodes.disk.geometry}>
      <shaderMaterial
        fragmentShader={Un.fragmentShader}
        vertexShader={Un.vertexShader}
        ref={shader}
        uniforms={uniform}
      />
    </mesh>
  );
};

useGLTF.preload('models//disc.glb');

function App() {
  return (
    <Canvas
      camera={{
        fov: 30,
        near: 0.1,
        far: 1000,
        position: [0, 0, 20],
      }}
    >
      <Disc />
      <OrbitControls />
    </Canvas>
  );
}

export default App;
