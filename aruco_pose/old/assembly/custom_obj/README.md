# Custom Assembly Assets

Place override assets under the fixed class directories in `assembly/custom_obj/`.

Supported directory names:

- `plate`
- `housing`
- `wedge`
- `node`
- `node_end` for the terminal node in the chain (with the default 10-node build, this is `node9`)
- `gripper_base`
- `gripper_claw`

Supported file names:

- Mesh OBJ: `<part_name>_mesh.obj`
- Frame JSON: `<part_name>_frame.json`
- Physics JSON: `<part_name>_physics.json`

Notes:

- Collision uses the same mesh path as visual.
- The builder only searches the fixed class directories above. It does not use per-instance folders such as `node1/`.
- The terminal node is a separate part kind, `node_end`, and searches only `node_end/`.
- Custom assets are used only when both `<part_name>_mesh.obj` and `<part_name>_frame.json` are present in the selected directory.
- If custom physics JSON is omitted, the builder estimates mass, center of mass, and inertia from the triangle mesh using a uniform density assumption.
- Even when a custom frame JSON is used, the builder rewrites its `connectors.from` and `connectors.to` fields from the assembly spec so assembly placement stays consistent.
