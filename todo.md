# Todo

- Two phase transport model : $\frac{\partial \phi_1}{\partial t} + \nabla \cdot (\mathbf{u}_1\,\phi_1) = s_1$ and $\frac{\partial \phi_2}{\partial t} + \nabla \cdot (\mathbf{u}_2\,\phi_2) = s_2$ with only 1 interface condition : flux continuity $\phi_1 \mathbf{u}_1 \cdot \mathbf{n} = \phi_2 \mathbf{u}_2 \cdot \mathbf{n}$ on $\Gamma$ (no scalar jump condition).
- Embedded interface inflow/outflow scalar BCs (currently only no-flow mode is supported)