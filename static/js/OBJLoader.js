/**
 * @author mrdoob / http://mrdoob.com/
 * @author Mugen87 / https://github.com/Mugen87
 */

THREE.OBJLoader = (function() {

	function OBJLoader(manager) {
		this.manager = manager !== undefined ? manager : THREE.DefaultLoadingManager;
		this.materials = null;
		this.path = '';
	}

	OBJLoader.prototype = {

		constructor: OBJLoader,

		load: function(url, onLoad, onProgress, onError) {
			const scope = this;
			const loader = new THREE.FileLoader(scope.manager);
			loader.setPath(scope.path);
			loader.load(url, function(text) {
				onLoad(scope.parse(text));
			}, onProgress, onError);
		},

		setPath: function(value) {
			this.path = value;
			return this;
		},

		setMaterials: function(materials) {
			this.materials = materials;
			return this;
		},

		parse: function(text) {
			console.time('OBJLoader');

			const object = new THREE.Group();
			object.name = '';

			const lines = text.split('\n');
			const vertices = [];
			const normals = [];
			const uvs = [];
			
			let geometry, position, normal, uv;

			for (let i = 0, l = lines.length; i < l; i++) {
				const line = lines[i].trim();

				if (line.length === 0 || line.charAt(0) === '#') continue;

				const elements = line.split(/\s+/);
				const command = elements[0];

				switch (command) {
					case 'v':
						vertices.push(
							parseFloat(elements[1]),
							parseFloat(elements[2]),
							parseFloat(elements[3])
						);
						break;

					case 'vt':
						uvs.push(
							parseFloat(elements[1]),
							1 - parseFloat(elements[2])
						);
						break;

					case 'vn':
						normals.push(
							parseFloat(elements[1]),
							parseFloat(elements[2]),
							parseFloat(elements[3])
						);
						break;

					case 'f':
						if (!geometry) {
							geometry = new THREE.BufferGeometry();
							position = [];
							normal = [];
							uv = [];
						}

						for (let j = 1, jl = elements.length; j < jl; j++) {
							const vertexData = elements[j].split('/');
							const vert = (vertexData[0] - 1) * 3;
							
							position.push(
								vertices[vert],
								vertices[vert + 1],
								vertices[vert + 2]
							);

							if (vertexData[1]) {
								const uvVert = (vertexData[1] - 1) * 2;
								uv.push(uvs[uvVert], uvs[uvVert + 1]);
							}

							if (vertexData[2]) {
								const normVert = (vertexData[2] - 1) * 3;
								normal.push(
									normals[normVert],
									normals[normVert + 1],
									normals[normVert + 2]
								);
							}
						}
						break;
				}
			}

			if (geometry) {
				geometry.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
				if (uv.length > 0) geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
				if (normal.length > 0) geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normal, 3));

				const material = new THREE.MeshStandardMaterial({ color: 0xffffff });
				const mesh = new THREE.Mesh(geometry, material);
				object.add(mesh);
			}

			console.timeEnd('OBJLoader');
			return object;
		}
	};

	return OBJLoader;

})();