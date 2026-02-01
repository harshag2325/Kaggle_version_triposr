// Minimal OrbitControls-like implementation for THREE (mouse rotate + zoom)
// Usage: controls = new THREE.OrbitControls(camera, renderer.domElement)

THREE.OrbitControls = function (object, domElement) {
  const scope = this;

  this.object = object;
  this.domElement = domElement || document;

  // public API
  this.enabled = true;
  this.target = new THREE.Vector3(0, 0, 0);

  this.minDistance = 0.1;
  this.maxDistance = Infinity;

  this.minPolarAngle = 0;               // radians
  this.maxPolarAngle = Math.PI;         // radians

  this.enableDamping = true;
  this.dampingFactor = 0.1;

  this.rotateSpeed = 1.0;
  this.zoomSpeed = 1.0;

  // Enable pan/rotate/zoom
  this.enableRotate = true;
  this.enableZoom = true;
  this.enablePan = true;

  // internal state
  const spherical = new THREE.Spherical();
  const sphericalDelta = new THREE.Spherical(0, 0, 0);
  const panOffset = new THREE.Vector3();
  const offset = new THREE.Vector3();

  let scale = 1;
  let isDragging = false;
  let rotateStart = new THREE.Vector2();
  let rotateEnd = new THREE.Vector2();
  let rotateDelta = new THREE.Vector2();

  // init spherical from current camera position
  offset.copy(scope.object.position).sub(scope.target);
  spherical.setFromVector3(offset);

  // Cache for calculations
  const quat = new THREE.Quaternion().setFromUnitVectors(
    object.up, 
    new THREE.Vector3(0, 1, 0)
  );
  const quatInverse = quat.clone().invert();

  function handleMouseDown(event) {
    if (!scope.enabled) return;
    event.preventDefault();

    isDragging = true;
    rotateStart.set(event.clientX, event.clientY);
    
    if (scope.domElement.setPointerCapture) {
      scope.domElement.setPointerCapture(event.pointerId);
    }
  }

  function handleMouseMove(event) {
    if (!scope.enabled || !isDragging || !scope.enableRotate) return;

    rotateEnd.set(event.clientX, event.clientY);
    rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(scope.rotateSpeed);
    rotateStart.copy(rotateEnd);

    const element = scope.domElement === document
      ? scope.domElement.body
      : scope.domElement;

    // Full 360° horizontal rotation (no limits)
    sphericalDelta.theta -= (2 * Math.PI * rotateDelta.x) / element.clientWidth;

    // Full 360° vertical rotation (no limits) - key change
    sphericalDelta.phi -= (2 * Math.PI * rotateDelta.y) / element.clientHeight;
  }

  function handleMouseUp(event) {
    if (!scope.enabled) return;
    isDragging = false;
    
    if (scope.domElement.releasePointerCapture) {
      scope.domElement.releasePointerCapture(event.pointerId);
    }
  }

  function handleWheel(event) {
    if (!scope.enabled || !scope.enableZoom) return;
    event.preventDefault();

    const zoomScale = Math.pow(0.95, scope.zoomSpeed);

    if (event.deltaY < 0) {
      scale /= zoomScale;
    } else if (event.deltaY > 0) {
      scale *= zoomScale;
    }
  }

  // event listeners
  this.domElement.addEventListener("pointerdown", handleMouseDown);
  this.domElement.addEventListener("pointermove", handleMouseMove);
  this.domElement.addEventListener("pointerup", handleMouseUp);
  this.domElement.addEventListener("pointercancel", handleMouseUp);
  this.domElement.addEventListener("wheel", handleWheel, { passive: false });

  // required public method (called in editor.js animation loop)
  this.update = function () {
    if (!scope.enabled) return;

    // apply rotation deltas
    spherical.theta += sphericalDelta.theta;
    spherical.phi += sphericalDelta.phi;

    // NO clamping for full 360° freedom - removed minPolarAngle/maxPolarAngle limits
    // If you want gimbal lock prevention, uncomment these:
    // spherical.phi = Math.max(0.000001, Math.min(Math.PI - 0.000001, spherical.phi));

    // Normalize theta to prevent float overflow
    spherical.theta = spherical.theta % (2 * Math.PI);
    
    // Allow phi to wrap for continuous rotation
    spherical.phi = spherical.phi % (2 * Math.PI);

    // apply zoom
    spherical.radius *= scale;

    // clamp distance
    spherical.radius = Math.max(
      scope.minDistance,
      Math.min(scope.maxDistance, spherical.radius)
    );

    // damping
    if (scope.enableDamping) {
      sphericalDelta.theta *= (1 - scope.dampingFactor);
      sphericalDelta.phi *= (1 - scope.dampingFactor);
    } else {
      sphericalDelta.set(0, 0, 0);
    }

    scale = 1;

    // apply pan offset
    scope.target.add(panOffset);
    panOffset.set(0, 0, 0);

    // convert back to Cartesian and update camera
    offset.setFromSpherical(spherical);

    // rotate offset back to "camera-up-vector-is-up" space
    offset.applyQuaternion(quatInverse);

    scope.object.position.copy(scope.target).add(offset);
    scope.object.lookAt(scope.target);
  };

  this.dispose = function () {
    scope.domElement.removeEventListener("pointerdown", handleMouseDown);
    scope.domElement.removeEventListener("pointermove", handleMouseMove);
    scope.domElement.removeEventListener("pointerup", handleMouseUp);
    scope.domElement.removeEventListener("pointercancel", handleMouseUp);
    scope.domElement.removeEventListener("wheel", handleWheel);
  };

  this.reset = function () {
    scope.target.set(0, 0, 0);
    offset.copy(scope.object.position).sub(scope.target);
    spherical.setFromVector3(offset);
    sphericalDelta.set(0, 0, 0);
    panOffset.set(0, 0, 0);
    scale = 1;
  };
};