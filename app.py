import base64
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageOps

from configs.infer_config import get_parser

video_examples = [
    ['test/videos/2.mp4', 1, -2, 1],
    ['test/videos/0-NNvgaTcVzAG0-r.mp4', 1, 5, 1],
    ['test/videos/9.mp4', 1, -2, 1],
    ['test/videos/p7.mp4', 1, -2, 1],
    ['test/videos/UST-fn-RvhJwMR5S.mp4', 1, -2, 1],
    ['test/videos/1.mp4', 1, -2, 1],
    ['test/videos/3.mp4', 1, -2, 1],
    ['test/videos/4.mp4', 1, -2, 1],
    # ['test/videos/5.mp4', 1, -2, 1],
    # ['test/videos/6.mp4', 1, -2, 1],
    # ['test/videos/7.mp4', 1, -2, 1],
    # ['test/videos/8.mp4', 1, -2, 1],
    # ['test/videos/10.mp4', 1, -2, 1],
    ['test/videos/ori1.mp4', 1, -2, 1],
    ['test/videos/part-2-3.mp4', 1, -2, 1],

]


_IMG_EXAMPLES_DIR = Path(__file__).resolve().parent / "test/images"
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_img_example_paths = []
if _IMG_EXAMPLES_DIR.exists():
    _img_example_paths = sorted(
        [p for p in _IMG_EXAMPLES_DIR.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS]
    )

# Gradio Examples format: [image_path, elevation, motion_radius_scale]
_ZERO_ELEVATION_EXAMPLES = {
    "seva_robot.png",
    "seva_yellow_dragon.png",
    "cat.png",
    "iron-elevation-n5.jpg",
}
img_examples = [[str(p), 0 if p.name in _ZERO_ELEVATION_EXAMPLES else 5, 1.0] for p in _img_example_paths]

max_seed = 2 ** 31


parser = get_parser() # infer config.py
opts = parser.parse_args() # default device: 'cuda:0'
prefix = datetime.now().strftime("%Y%m%d_%H%M")
opts.save_dir = f'./output/gradio/{prefix}'
os.makedirs(opts.save_dir,exist_ok=True)
# Keep `opts.device` as passed from CLI (e.g. "cuda:0" or "cpu").
opts.weight_dtype = torch.bfloat16

from demo import UniScene


APP_CSS = r"""
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }

.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3 {
  font-family: "Fraunces", ui-serif, Georgia, serif;
  letter-spacing: -0.02em;
}
.gradio-container, .gradio-container label, .gradio-container button, .gradio-container input, .gradio-container textarea {
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}

body {
  background:
    radial-gradient(1100px 600px at 10% -20%, rgba(0, 255, 136, 0.14), transparent 55%),
    radial-gradient(900px 520px at 90% 0%, rgba(255, 105, 180, 0.14), transparent 55%),
    radial-gradient(700px 500px at 60% 110%, rgba(255, 165, 0, 0.12), transparent 60%),
    #0b0e14;
}

#uv-hero {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 18px 18px 14px;
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
  backdrop-filter: blur(10px);
}

.uv-subtle {
  color: rgba(255,255,255,0.78);
  line-height: 1.35;
}

.gradio-container .gr-button-primary {
  background: linear-gradient(135deg, rgba(0,255,136,0.92), rgba(255,105,180,0.92));
  border: none !important;
  color: #0b0e14 !important;
  font-weight: 700;
}

.gradio-container .gr-button-primary:hover {
  filter: brightness(1.02);
}

.uv-camera-control-wrapper { box-shadow: 0 18px 60px rgba(0,0,0,0.40); }
"""

APP_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
"""

CAMERA_3D_HTML_TEMPLATE = """
<div class="uv-camera-control-wrapper" style="width: 100%; height: 560px; position: relative; background: #111319; border-radius: 18px; overflow: hidden; touch-action: none;">
    <div class="uv-prompt-overlay" style="position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.78); padding: 10px 16px; border-radius: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #00ff88; white-space: nowrap; z-index: 10; max-width: 92%; overflow: hidden; text-overflow: ellipsis;"></div>
    <div class="uv-control-legend" style="position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.60); padding: 12px 14px; border-radius: 14px; font-family: system-ui; font-size: 12px; color: rgba(255,255,255,0.90); z-index: 10;">
        <div style="margin-bottom: 6px;"><span style="color: #00ff88;">●</span> Yaw / Orbit (d_phi)</div>
        <div style="margin-bottom: 6px;"><span style="color: #ff69b4;">●</span> Pitch / Tilt (d_theta)</div>
        <div><span style="color: #ffa500;">●</span> Dolly (z_offset)</div>
    </div>
</div>
"""

CAMERA_3D_JS = r"""
(() => {
  const wrapper = element.querySelector('.uv-camera-control-wrapper');
  const promptOverlay = element.querySelector('.uv-prompt-overlay');

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const snap = (v, step) => Math.round(v / step) * step;
  const degToRad = (d) => (d * Math.PI) / 180.0;
  const radToDeg = (r) => (r * 180.0) / Math.PI;
  const unwrapDeltaDeg = (d) => ((d + 540) % 360) - 180;

  const initScene = () => {
    if (typeof THREE === 'undefined') {
      setTimeout(initScene, 100);
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111319);

    const camera = new THREE.PerspectiveCamera(55, wrapper.clientWidth / wrapper.clientHeight, 0.1, 1000);
    camera.position.set(4.6, 3.4, 4.6);
    camera.lookAt(0, 0.75, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
    const dpr = Math.min(Math.max(window.devicePixelRatio || 1, 2), 3);
    renderer.setPixelRatio(dpr);
    wrapper.insertBefore(renderer.domElement, wrapper.firstChild);

    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.55);
    dirLight.position.set(6, 10, 5);
    scene.add(dirLight);

	    scene.add(new THREE.GridHelper(6, 12, 0x2a2d36, 0x1b1e26));
	
	    const CENTER = new THREE.Vector3(0, 0.75, 0);
	    const PHI_MIN = (props.phiMin ?? -360);
	    const PHI_MAX = (props.phiMax ?? 360);
	    const THETA_MIN = -30;
	    const THETA_MAX = 30;
	    const Z_MIN = -0.2;
	    const Z_MAX = 0.2;
	    const BASE_DISTANCE = 2.2;
	    const DIST_SCALE = 1.05;
	    const ROTATION_RADIUS = 2.05;
	    const TILT_RADIUS = 1.75;
	    const PITCH_ARC_ANGLE_MIN = -THETA_MAX; // -30
	    const PITCH_ARC_ANGLE_MAX = -THETA_MIN; // 30

    const ARC_TUBE_RADIUS = 0.04;
    const ARC_RADIAL_SEGMENTS = 24;
    const ORBIT_POINTS = 256;
    const ORBIT_TUBULAR_SEGMENTS = 256;
    const PITCH_POINTS = 180;
    const PITCH_TUBULAR_SEGMENTS = 180;
    const HANDLE_SEGMENTS = 28;

	    let dPhi = props.value?.d_phi ?? 0;
	    let dTheta = props.value?.d_theta ?? 0;
	    let zOffset = props.value?.z_offset ?? 0;

	    dPhi = clamp(dPhi, PHI_MIN, PHI_MAX);
	    dTheta = clamp(dTheta, THETA_MIN, THETA_MAX);
	    zOffset = clamp(zOffset, Z_MIN, Z_MAX);

    function createPlaceholderTexture() {
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 256;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#2b2f3a';
      ctx.fillRect(0, 0, 256, 256);
      ctx.fillStyle = '#00ff88';
      ctx.globalAlpha = 0.12;
      ctx.beginPath();
      ctx.arc(128, 128, 90, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = '#f4f6ff';
      ctx.font = '700 18px system-ui';
      ctx.fillText('UniView', 92, 128);
      return new THREE.CanvasTexture(canvas);
    }

    const planeMaterial = new THREE.MeshBasicMaterial({ map: createPlaceholderTexture(), side: THREE.DoubleSide });
    let targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.25, 1.25), planeMaterial);
    targetPlane.position.copy(CENTER);
    scene.add(targetPlane);

    function updateTextureFromUrl(url) {
      if (!url) {
        planeMaterial.map = createPlaceholderTexture();
        planeMaterial.needsUpdate = true;
        return;
      }
      const loader = new THREE.TextureLoader();
      loader.crossOrigin = 'anonymous';
      loader.load(url, (texture) => {
        texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
        texture.generateMipmaps = true;
        texture.minFilter = THREE.LinearMipmapLinearFilter;
        texture.magFilter = THREE.LinearFilter;
        planeMaterial.map = texture;
        planeMaterial.needsUpdate = true;
      });
    }

    if (props.imageUrl) updateTextureFromUrl(props.imageUrl);

    const cameraGroup = new THREE.Group();
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x6aa7ff, metalness: 0.45, roughness: 0.25 });
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.2, 0.35), bodyMat);
    cameraGroup.add(body);
    const lens = new THREE.Mesh(
      new THREE.CylinderGeometry(0.08, 0.1, 0.16, 24),
      new THREE.MeshStandardMaterial({ color: 0x6aa7ff, metalness: 0.45, roughness: 0.25 })
    );
    lens.rotation.x = Math.PI / 2;
    lens.position.z = 0.24;
    cameraGroup.add(lens);
    scene.add(cameraGroup);

    const rotationArcPoints = [];
    for (let i = 0; i <= ORBIT_POINTS; i++) {
      const angle = THREE.MathUtils.degToRad(-180 + (360 * i / ORBIT_POINTS));
      rotationArcPoints.push(new THREE.Vector3(ROTATION_RADIUS * Math.sin(angle), 0.05, ROTATION_RADIUS * Math.cos(angle)));
    }
    const rotationCurve = new THREE.CatmullRomCurve3(rotationArcPoints);
    const rotationArc = new THREE.Mesh(
      new THREE.TubeGeometry(rotationCurve, ORBIT_TUBULAR_SEGMENTS, ARC_TUBE_RADIUS, ARC_RADIAL_SEGMENTS, true),
      new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.22 })
    );
    scene.add(rotationArc);

    const rotationHandle = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, HANDLE_SEGMENTS, HANDLE_SEGMENTS),
      new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.55 })
    );
    rotationHandle.userData.type = 'yaw';
    scene.add(rotationHandle);

    const tiltArcPoints = [];
    for (let i = 0; i <= PITCH_POINTS; i++) {
      const angleDeg = PITCH_ARC_ANGLE_MIN + ((PITCH_ARC_ANGLE_MAX - PITCH_ARC_ANGLE_MIN) * i / PITCH_POINTS);
      const angle = THREE.MathUtils.degToRad(angleDeg);
      tiltArcPoints.push(new THREE.Vector3(-0.7, TILT_RADIUS * Math.sin(angle) + CENTER.y, TILT_RADIUS * Math.cos(angle)));
    }
    const tiltCurve = new THREE.CatmullRomCurve3(tiltArcPoints);
    const tiltArc = new THREE.Mesh(
      new THREE.TubeGeometry(tiltCurve, PITCH_TUBULAR_SEGMENTS, ARC_TUBE_RADIUS, ARC_RADIAL_SEGMENTS, false),
      new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.22 })
    );
    scene.add(tiltArc);

    const tiltHandle = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, HANDLE_SEGMENTS, HANDLE_SEGMENTS),
      new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.55 })
    );
    tiltHandle.userData.type = 'pitch';
    scene.add(tiltHandle);

    const distanceLineGeo = new THREE.BufferGeometry();
    const distanceLine = new THREE.Line(distanceLineGeo, new THREE.LineBasicMaterial({ color: 0xffa500 }));
    scene.add(distanceLine);

    const distanceHandle = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, HANDLE_SEGMENTS, HANDLE_SEGMENTS),
      new THREE.MeshStandardMaterial({ color: 0xffa500, emissive: 0xffa500, emissiveIntensity: 0.55 })
    );
    distanceHandle.userData.type = 'dolly';
    scene.add(distanceHandle);

    function overlayText(phi, theta, z) {
      const sPhi = (phi >= 0 ? '+' : '') + phi.toFixed(0) + '°';
      const sTheta = (theta >= 0 ? '+' : '') + theta.toFixed(0) + '°';
      const sZ = (z >= 0 ? '+' : '') + z.toFixed(2);
      return `Yaw d_phi: ${sPhi}  •  Pitch d_theta: ${sTheta}  •  Dolly z_offset: ${sZ}`;
    }

	    function updatePositions() {
	      dPhi = clamp(dPhi, PHI_MIN, PHI_MAX);
	      dTheta = clamp(dTheta, THETA_MIN, THETA_MAX);
	      zOffset = clamp(zOffset, Z_MIN, Z_MAX);

      const rotRad = degToRad(-dPhi);
      const tiltRad = degToRad(-dTheta);
      const distance = clamp(BASE_DISTANCE - (zOffset * DIST_SCALE), 0.9, 4.2);

      const camX = distance * Math.sin(rotRad) * Math.cos(tiltRad);
      const camY = distance * Math.sin(tiltRad) + CENTER.y;
      const camZ = distance * Math.cos(rotRad) * Math.cos(tiltRad);

      cameraGroup.position.set(camX, camY, camZ);
      cameraGroup.lookAt(CENTER);

      rotationHandle.position.set(ROTATION_RADIUS * Math.sin(rotRad), 0.05, ROTATION_RADIUS * Math.cos(rotRad));

      const tiltHandleAngle = degToRad(-dTheta);
      tiltHandle.position.set(-0.7, TILT_RADIUS * Math.sin(tiltHandleAngle) + CENTER.y, TILT_RADIUS * Math.cos(tiltHandleAngle));

      const handleDist = Math.max(0.6, distance - 0.45);
      distanceHandle.position.set(
        handleDist * Math.sin(rotRad) * Math.cos(tiltRad),
        handleDist * Math.sin(tiltRad) + CENTER.y,
        handleDist * Math.cos(rotRad) * Math.cos(tiltRad)
      );
      distanceLineGeo.setFromPoints([cameraGroup.position.clone(), CENTER.clone()]);

      promptOverlay.textContent = overlayText(dPhi, dTheta, zOffset);
    }

	    function updatePropsAndTrigger() {
	      const phiSnap = snap(dPhi, 1);
	      const thetaSnap = snap(dTheta, 1);
	      const zSnap = snap(zOffset, 0.01);
	      props.value = { d_phi: phiSnap, d_theta: thetaSnap, z_offset: zSnap };
      trigger('change', props.value);
    }

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const intersection = new THREE.Vector3();
    let isDragging = false;
    let dragTarget = null;

    let startZ = 0;
    let lastYawAngle = 0;
    let activePointerId = null;
    let dollyCenterPx = new THREE.Vector2();
    let startMousePx = new THREE.Vector2();
    let currMousePx = new THREE.Vector2();
    let startDollyRadius = 0;
    const DOLLY_SENS_PX = 0.0065;

    const canvas = renderer.domElement;

    function setMouseFromEvent(e) {
      const rect = canvas.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    }

    function intersectYawAngle() {
      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
      if (!raycaster.ray.intersectPlane(plane, intersection)) return null;
      return radToDeg(Math.atan2(intersection.x, intersection.z));
    }

    canvas.addEventListener('pointerdown', (e) => {
      setMouseFromEvent(e);
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects([rotationHandle, tiltHandle, distanceHandle]);
      if (hits.length === 0) return;

      try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
      activePointerId = e.pointerId;
      isDragging = true;
      dragTarget = hits[0].object;
      dragTarget.material.emissiveIntensity = 1.0;
      dragTarget.scale.setScalar(1.3);
      startZ = zOffset;

      if (dragTarget.userData.type === 'yaw') {
        const a = intersectYawAngle();
        lastYawAngle = a == null ? lastYawAngle : a;
      }
      if (dragTarget.userData.type === 'dolly') {
        const rect = canvas.getBoundingClientRect();
        const centerNdc = CENTER.clone().project(camera);
        dollyCenterPx.set(
          (centerNdc.x + 1) * 0.5 * rect.width,
          (1 - (centerNdc.y + 1) * 0.5) * rect.height
        );
        startMousePx.set(e.clientX - rect.left, e.clientY - rect.top);
        startDollyRadius = Math.max(1, startMousePx.distanceTo(dollyCenterPx));
      }

      canvas.style.cursor = 'grabbing';
    });

    canvas.addEventListener('pointermove', (e) => {
      if (activePointerId != null && e.pointerId !== activePointerId) return;
      setMouseFromEvent(e);

      if (isDragging && dragTarget) {
        raycaster.setFromCamera(mouse, camera);

	        if (dragTarget.userData.type === 'yaw') {
	          const a = intersectYawAngle();
	          if (a != null) {
	            const delta = unwrapDeltaDeg(a - lastYawAngle);
	            dPhi = clamp(dPhi - delta, PHI_MIN, PHI_MAX);
	            lastYawAngle = a;
	          }
	        } else if (dragTarget.userData.type === 'pitch') {
          const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0.7);
          if (raycaster.ray.intersectPlane(plane, intersection)) {
            const relY = intersection.y - CENTER.y;
            const relZ = intersection.z;
            const angle = radToDeg(Math.atan2(relY, relZ)); // [-90..90]
            const clampedAngle = clamp(angle, PITCH_ARC_ANGLE_MIN, PITCH_ARC_ANGLE_MAX);
            dTheta = clamp(-clampedAngle, THETA_MIN, THETA_MAX);
          }
	        } else if (dragTarget.userData.type === 'dolly') {
	          const rect = canvas.getBoundingClientRect();
	          currMousePx.set(e.clientX - rect.left, e.clientY - rect.top);
	          const currR = currMousePx.distanceTo(dollyCenterPx);
	          const deltaR = currR - startDollyRadius;
	          zOffset = clamp(startZ - deltaR * DOLLY_SENS_PX, Z_MIN, Z_MAX);
	        }

        updatePositions();
      } else {
        raycaster.setFromCamera(mouse, camera);
        const hits = raycaster.intersectObjects([rotationHandle, tiltHandle, distanceHandle]);
        [rotationHandle, tiltHandle, distanceHandle].forEach((h) => {
          h.material.emissiveIntensity = 0.55;
          h.scale.setScalar(1);
        });
        if (hits.length > 0) {
          hits[0].object.material.emissiveIntensity = 0.85;
          hits[0].object.scale.setScalar(1.12);
          canvas.style.cursor = 'grab';
        } else {
          canvas.style.cursor = 'default';
        }
      }
    });

    const onPointerUp = (e) => {
      if (activePointerId != null && e.pointerId !== activePointerId) return;
      if (dragTarget) {
        dragTarget.material.emissiveIntensity = 0.55;
        dragTarget.scale.setScalar(1);
        updatePropsAndTrigger();
      }
      isDragging = false;
      dragTarget = null;
      if (activePointerId != null) {
        try { canvas.releasePointerCapture(activePointerId); } catch (_) {}
      }
      activePointerId = null;
      canvas.style.cursor = 'default';
    };

    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointercancel', onPointerUp);

    updatePositions();

    function render() {
      requestAnimationFrame(render);
      renderer.render(scene, camera);
    }
    render();

    new ResizeObserver(() => {
      camera.aspect = wrapper.clientWidth / wrapper.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
    }).observe(wrapper);

    // Sync from python -> JS when props.value changes
    let lastValue = JSON.stringify(props.value);
    let lastImageUrl = props.imageUrl;
    setInterval(() => {
      const currentValue = JSON.stringify(props.value);
      if (currentValue !== lastValue) {
        lastValue = currentValue;
	        if (props.value && typeof props.value === 'object') {
	          dPhi = clamp(props.value.d_phi ?? dPhi, PHI_MIN, PHI_MAX);
	          dTheta = clamp(props.value.d_theta ?? dTheta, THETA_MIN, THETA_MAX);
	          zOffset = clamp(props.value.z_offset ?? zOffset, Z_MIN, Z_MAX);
	          updatePositions();
	        }
	      }
      if (props.imageUrl !== lastImageUrl) {
        lastImageUrl = props.imageUrl;
        updateTextureFromUrl(props.imageUrl);
      }
    }, 100);
  };

  initScene();
})();
	"""

def create_camera_3d_component(value=None, imageUrl=None, phiMin=None, phiMax=None, **kwargs):
    if value is None:
        value = {"d_phi": 0.0, "d_theta": 0.0, "z_offset": 0.0}
    return gr.HTML(
        value=value,
        html_template=CAMERA_3D_HTML_TEMPLATE,
        js_on_load=CAMERA_3D_JS,
        imageUrl=imageUrl,
        phiMin=phiMin,
        phiMax=phiMax,
        **kwargs,
    )

def _camera_3d_widget_html(widget_id: str, dphi_elem_id: str, dtheta_elem_id: str, z_elem_id: str) -> str:
    raise RuntimeError("Legacy Gradio3 widget should not be used under Gradio 6.")


def _normalize_checkbox(value) -> bool:
    if isinstance(value, str):
        value = value.lower() in {"true", "1", "yes", "y", "on"}
    elif isinstance(value, (int, float)):
        value = bool(value)
    return bool(value)


def _pose_from_values(d_phi: float, d_theta: float, z_offset: float) -> str:
    return f"{float(d_phi):.3f}; {float(d_theta):.3f}; 0; 0; {float(z_offset):.3f}"


def _update_pose(d_phi: float, d_theta: float, z_offset: float) -> str:
    return _pose_from_values(d_phi, d_theta, z_offset)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


_D_PHI_MIN, _D_PHI_MAX = -360.0, 360.0
_D_THETA_MIN, _D_THETA_MAX = -30.0, 30.0
_Z_OFFSET_MIN, _Z_OFFSET_MAX = -0.2, 0.2


def _sync_3d_to_controls(camera_value: dict):
    if camera_value and isinstance(camera_value, dict):
        d_phi = float(camera_value.get("d_phi", 0.0) or 0.0)
        d_theta = float(camera_value.get("d_theta", 0.0) or 0.0)
        z_offset = float(camera_value.get("z_offset", 0.0) or 0.0)
    else:
        d_phi, d_theta, z_offset = 0.0, 0.0, 0.0
    d_phi = _clamp(d_phi, _D_PHI_MIN, _D_PHI_MAX)
    d_theta = _clamp(d_theta, _D_THETA_MIN, _D_THETA_MAX)
    z_offset = _clamp(z_offset, _Z_OFFSET_MIN, _Z_OFFSET_MAX)
    pose = _pose_from_values(d_phi, d_theta, z_offset)
    return d_phi, d_theta, z_offset, pose


def _sync_controls_to_3d(d_phi: float, d_theta: float, z_offset: float):
    d_phi = _clamp(float(d_phi), _D_PHI_MIN, _D_PHI_MAX)
    d_theta = _clamp(float(d_theta), _D_THETA_MIN, _D_THETA_MAX)
    z_offset = _clamp(float(z_offset), _Z_OFFSET_MIN, _Z_OFFSET_MAX)
    cam = {"d_phi": d_phi, "d_theta": d_theta, "z_offset": z_offset}
    pose = _pose_from_values(d_phi, d_theta, z_offset)
    return cam, pose


def _update_3d_image(img_np):
    if img_np is None:
        return gr.update(imageUrl=None)
    img = Image.fromarray(img_np).convert("RGB")
    img = ImageOps.exif_transpose(img)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    data_url = f"data:image/png;base64,{img_str}"
    return gr.update(imageUrl=data_url)


def _gradio_video_to_path(video_value) -> str | None:
    if video_value is None:
        return None
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, (tuple, list)) and video_value and isinstance(video_value[0], str):
        return video_value[0]
    if isinstance(video_value, dict):
        for key in ("path", "name", "file", "video"):
            value = video_value.get(key)
            if isinstance(value, str):
                return value
    return None


def _update_3d_image_from_video(video_value):
    video_path = _gradio_video_to_path(video_value)
    if not video_path or not os.path.exists(video_path):
        return gr.update(imageUrl=None)
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        if len(vr) <= 0:
            return gr.update(imageUrl=None)
        first_frame = vr[0].asnumpy()  # RGB uint8 HWC
        return _update_3d_image(first_frame)
    except Exception:
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            ok, frame_bgr = cap.read()
            cap.release()
            if not ok or frame_bgr is None:
                return gr.update(imageUrl=None)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return _update_3d_image(frame_rgb)
        except Exception:
            return gr.update(imageUrl=None)


def uniscene_demo(opts):
    # init model
    image2video = UniScene(opts, gradio=True)

    with gr.Blocks(analytics_enabled=False) as uniscene_iface:
        with gr.Column(elem_id="uv-hero"):
            gr.Markdown(
                """
                # UniView: Large-Baseline View Synthesis via Video Diffusion Models
                <div class="uv-subtle">
                A geometry-aware NVS demo (STream3R + depth alignment + video diffusion).<br/>
                Control the camera using the 3D widget or sliders.<br/>
                <a href="https://github.com/PKU-YuanGroup/UniView" target="_blank" rel="noopener noreferrer">GitHub: PKU-YuanGroup/UniView</a>
                </div>
                """,
                elem_id="uv-title",
            )

        def run_single_view_handler(
            image,
            elevation,
            center_scale,
            d_phi,
            d_theta,
            z_offset,
            steps,
            seed,
            guidance,
            render_method,
            occlusion_enlarge,
            keep_aspect_ratio,
            vis_threshold,
        ):
            enabled = _normalize_checkbox(occlusion_enlarge)
            image2video.opts.keep_aspect_ratio = _normalize_checkbox(keep_aspect_ratio)
            rm = str(render_method).strip().lower()
            image2video.opts.render_method = rm
            image2video.opts.warp_with_occlusion = enabled if rm == "warp" else False
            pose = _pose_from_values(float(d_phi), float(d_theta), float(z_offset))
            return image2video.run_single_view_gradio(
                image,
                elevation,
                center_scale,
                pose,
                steps,
                seed,
                guidance,
                vis_threshold,
            )

        def run_dynamic_view_handler(
            video,
            stride,
            elevation,
            center_scale,
            d_phi,
            d_theta,
            z_offset,
            steps,
            seed,
            guidance,
            render_method,
            align_with_vda,
            warp_with_occlusion,
            keep_aspect_ratio,
            vis_threshold,
        ):
            rm = str(render_method).strip().lower()
            align_enabled = _normalize_checkbox(align_with_vda)
            occl_enabled = _normalize_checkbox(warp_with_occlusion)
            image2video.opts.keep_aspect_ratio = _normalize_checkbox(keep_aspect_ratio)

            image2video.opts.render_method = rm
            image2video.opts.align_with_vda = align_enabled
            if rm == "warp":
                image2video.opts.warp_with_occlusion = occl_enabled
                image2video.opts.advanced_render = align_enabled or occl_enabled
            else:
                image2video.opts.warp_with_occlusion = False
                image2video.opts.advanced_render = align_enabled

            pose = _pose_from_values(float(d_phi), float(d_theta), float(z_offset))
            return image2video.run_dynamic_view_gradio(
                video,
                stride,
                elevation,
                center_scale,
                pose,
                steps,
                seed,
                guidance,
                vis_threshold,
            )

        with gr.Tabs():
            with gr.TabItem("🖼️ Image → Novel View Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        i2v_input_image = gr.Image(label="Input image", type="numpy", height=280)

                        with gr.Tab("🎮 3D camera control"):
                            camera_3d_single = create_camera_3d_component(
                                value={"d_phi": 0.0, "d_theta": 0.0, "z_offset": 0.0},
                                imageUrl=None,
                                phiMin=-360,
                                phiMax=360,
                            )

                        with gr.Tab("🎚️ Slider controls"):
                            d_phi_single = gr.Slider(
                                elem_id="d_phi_single",
                                label="Yaw / orbit (d_phi) — right (-) / left (+) [deg]",
                                minimum=-360,
                                maximum=360,
                                step=1,
                                value=0,
                            )
                            d_theta_single = gr.Slider(
                                elem_id="d_theta_single",
                                label="Pitch / tilt (d_theta) — up (-) / down (+) [deg]",
                                minimum=-30,
                                maximum=30,
                                step=1,
                                value=0,
                            )
                            z_offset_single = gr.Slider(
                                elem_id="z_offset_single",
                                label="Dolly (z_offset) — back (-) / forward (+)",
                                minimum=-0.2,
                                maximum=0.2,
                                step=0.01,
                                value=0.0,
                            )

                        pose_single = gr.Textbox(
                            label="Pose string (d_phi; d_theta; x; y; z)",
                            value="0; 0; 0; 0; 0",
                            interactive=False,
                            visible=False,
                        )

                    with gr.Column(scale=1):
                        i2v_output_video = gr.Video(
                            label="Output video",
                            autoplay=True,
                            buttons=["download"],
                        )

                        with gr.Row():
                            i2v_elevation = gr.Slider(
                                minimum=-45,
                                maximum=45,
                                step=1,
                                label="Initial elevation [deg]",
                                value=5,
                            )
                            i2v_center_scale = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                step=0.1,
                                label="Motion radius scale",
                                value=1.0,
                            )

                        with gr.Accordion("Advanced settings", open=False):
                            i2v_steps = gr.Slider(
                                minimum=4,
                                maximum=10,
                                step=1,
                                label="Denoising steps",
                                value=8,
                            )
                            i2v_seed = gr.Slider(
                                label="Seed",
                                minimum=0,
                                maximum=max_seed,
                                step=1,
                                value=0,
                            )
                            i2v_guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=8.0,
                                step=0.5,
                                label="Guidance scale",
                                value=4.0,
                            )
                            render_method_single = gr.Radio(
                                choices=["hybrid", "mesh", "warp"],
                                value="warp",
                                label="Render method",
                                interactive=True,
                            )
                            occlusion_enlarge_single = gr.Checkbox(
                                value=True,
                                label="Occlusion-aware warping (warp only)",
                            )
                            keep_aspect_ratio_single = gr.Checkbox(
                                value=True,
                                label="Keep input aspect ratio (no pad / no crop; ~480×832 token budget)",
                            )
                            vis_threshold_single = gr.Slider(
                                minimum=-1.0,
                                maximum=1.0,
                                step=0.05,
                                value=-0.5,
                                label="Visibility threshold",
                            )

                        i2v_end_btn = gr.Button("Generate", variant="primary")

                gr.Examples(
                    examples=img_examples,
                    inputs=[i2v_input_image, i2v_elevation, i2v_center_scale],
                    label="Examples",
                )

                def _update_single_occlusion_visibility(method):
                    m = str(method).strip().lower()
                    if m in {"hybrid", "mesh"}:
                        return gr.update(visible=False, interactive=False, value=False)
                    return gr.update(visible=True, interactive=True, value=True)

                render_method_single.change(
                    fn=_update_single_occlusion_visibility,
                    inputs=[render_method_single],
                    outputs=[occlusion_enlarge_single],
                )

                i2v_input_image.change(
                    fn=_update_3d_image,
                    inputs=[i2v_input_image],
                    outputs=[camera_3d_single],
                )

                camera_3d_single.change(
                    fn=_sync_3d_to_controls,
                    inputs=[camera_3d_single],
                    outputs=[d_phi_single, d_theta_single, z_offset_single, pose_single],
                )

                for s in (d_phi_single, d_theta_single, z_offset_single):
                    s.change(
                        fn=_sync_controls_to_3d,
                        inputs=[d_phi_single, d_theta_single, z_offset_single],
                        outputs=[camera_3d_single, pose_single],
                    )

                i2v_end_btn.click(
                    inputs=[
                        i2v_input_image,
                        i2v_elevation,
                        i2v_center_scale,
                        d_phi_single,
                        d_theta_single,
                        z_offset_single,
                        i2v_steps,
                        i2v_seed,
                        i2v_guidance_scale,
                        render_method_single,
                        occlusion_enlarge_single,
                        keep_aspect_ratio_single,
                        vis_threshold_single,
                    ],
                    outputs=[i2v_output_video],
                    fn=run_single_view_handler,
                )

            with gr.TabItem("🎞️ Video → Novel View Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        i2v_input_video = gr.Video(label="Input video", format="mp4", height=280)

                        i2v_stride = gr.Slider(
                            minimum=1,
                            maximum=5,
                            step=1,
                            label="Frame stride (sampling interval)",
                            value=1,
                        )

                        with gr.Tab("🎮 3D camera control"):
                            camera_3d_dyn = create_camera_3d_component(
                                value={"d_phi": 0.0, "d_theta": 0.0, "z_offset": 0.0},
                                imageUrl=None,
                                phiMin=-180,
                                phiMax=180,
                            )

                        with gr.Tab("🎚️ Slider controls"):
                            d_phi_dyn = gr.Slider(
                                elem_id="d_phi_dyn",
                                label="Yaw / orbit (d_phi) — right (-) / left (+) [deg]",
                                minimum=-180,
                                maximum=180,
                                step=1,
                                value=0,
                            )
                            d_theta_dyn = gr.Slider(
                                elem_id="d_theta_dyn",
                                label="Pitch / tilt (d_theta) — up (-) / down (+) [deg]",
                                minimum=-30,
                                maximum=30,
                                step=1,
                                value=0,
                            )
                            z_offset_dyn = gr.Slider(
                                elem_id="z_offset_dyn",
                                label="Dolly (z_offset) — back (-) / forward (+)",
                                minimum=-0.2,
                                maximum=0.2,
                                step=0.01,
                                value=0.0,
                            )

                        pose_dyn = gr.Textbox(
                            label="Pose string (d_phi; d_theta; x; y; z)",
                            value="0; 0; 0; 0; 0",
                            interactive=False,
                            visible=False,
                        )

                    with gr.Column(scale=1):
                        i2v_output_video_dyn = gr.Video(
                            label="Output video",
                            autoplay=True,
                            buttons=["download"],
                        )

                        with gr.Row():
                            i2v_elevation_dyn = gr.Slider(
                                minimum=-45,
                                maximum=45,
                                step=1,
                                label="Initial elevation [deg]",
                                value=-5,
                            )
                            i2v_center_scale_dyn = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                step=0.1,
                                label="Motion radius scale",
                                value=1.0,
                            )

                        with gr.Accordion("Advanced settings", open=False):
                            i2v_steps_dyn = gr.Slider(
                                minimum=4,
                                maximum=10,
                                step=1,
                                label="Denoising steps",
                                value=8,
                            )
                            i2v_seed_dyn = gr.Slider(
                                label="Seed",
                                minimum=0,
                                maximum=max_seed,
                                step=1,
                                value=0,
                            )
                            i2v_guidance_scale_dyn = gr.Slider(
                                minimum=1.0,
                                maximum=8.0,
                                step=0.5,
                                label="Guidance scale",
                                value=4.0,
                            )
                            render_method_toggle = gr.Radio(
                                choices=["hybrid", "mesh", "warp"],
                                value="hybrid",
                                label="Render method",
                                interactive=True,
                            )
                            align_with_vda_toggle = gr.Checkbox(
                                value=True,
                                label="Enable VDA depth alignment",
                            )
                            warp_with_occlusion_toggle = gr.Checkbox(
                                value=False,
                                label="Occlusion-aware warping (warp only)",
                                visible=False,
                                interactive=False,
                            )
                            keep_aspect_ratio_dyn = gr.Checkbox(
                                value=True,
                                label="Keep input aspect ratio (no pad / no crop; ~480×832 token budget)",
                            )
                            vis_threshold_dyn = gr.Slider(
                                minimum=-1.0,
                                maximum=1.0,
                                step=0.05,
                                value=-0.5,
                                label="Visibility threshold",
                            )

                        i2v_end_btn_dyn = gr.Button("Generate", variant="primary")

                gr.Examples(
                    examples=video_examples,
                    inputs=[i2v_input_video, i2v_stride, i2v_elevation_dyn, i2v_center_scale_dyn],
                    label="Examples",
                )

                i2v_input_video.change(
                    fn=_update_3d_image_from_video,
                    inputs=[i2v_input_video],
                    outputs=[camera_3d_dyn],
                )

                def _update_dynamic_occlusion_visibility(method):
                    m = str(method).strip().lower()
                    if m == "warp":
                        return gr.update(visible=True, interactive=True, value=True)
                    return gr.update(visible=False, interactive=False, value=False)

                render_method_toggle.change(
                    fn=_update_dynamic_occlusion_visibility,
                    inputs=[render_method_toggle],
                    outputs=[warp_with_occlusion_toggle],
                )

                camera_3d_dyn.change(
                    fn=_sync_3d_to_controls,
                    inputs=[camera_3d_dyn],
                    outputs=[d_phi_dyn, d_theta_dyn, z_offset_dyn, pose_dyn],
                )

                for s in (d_phi_dyn, d_theta_dyn, z_offset_dyn):
                    s.change(
                        fn=_sync_controls_to_3d,
                        inputs=[d_phi_dyn, d_theta_dyn, z_offset_dyn],
                        outputs=[camera_3d_dyn, pose_dyn],
                    )

                i2v_end_btn_dyn.click(
                    inputs=[
                        i2v_input_video,
                        i2v_stride,
                        i2v_elevation_dyn,
                        i2v_center_scale_dyn,
                        d_phi_dyn,
                        d_theta_dyn,
                        z_offset_dyn,
                        i2v_steps_dyn,
                        i2v_seed_dyn,
                        i2v_guidance_scale_dyn,
                        render_method_toggle,
                        align_with_vda_toggle,
                        warp_with_occlusion_toggle,
                        keep_aspect_ratio_dyn,
                        vis_threshold_dyn,
                    ],
                    outputs=[i2v_output_video_dyn],
                    fn=run_dynamic_view_handler,
                )

    return uniscene_iface


uniscene_iface = uniscene_demo(opts)
uniscene_iface.queue(max_size=10)
launch_kwargs = {
    "debug": True,
    "css": APP_CSS,
    "theme": gr.themes.Citrus(),
    "head": APP_HEAD,
    "share": False,
}
if os.environ.get("GRADIO_SERVER_NAME"):
    launch_kwargs["server_name"] = os.environ["GRADIO_SERVER_NAME"]
if os.environ.get("GRADIO_SERVER_PORT"):
    launch_kwargs["server_port"] = int(os.environ["GRADIO_SERVER_PORT"])

uniscene_iface.launch(**launch_kwargs)
