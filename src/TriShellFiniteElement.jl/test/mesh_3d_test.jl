
using Ferrite, FerriteGmsh

gmsh.initialize()

    # Add a unit sphere in 3D space
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
    gmsh.model.occ.synchronize()

    # Generate nodes and surface elements only, hence we need to pass 2 into generate
    gmsh.model.mesh.generate(2)

    # # To get good solution quality refine the elements several times
    # for _ in 1:refinements
    #     gmsh.model.mesh.refine()
    # end

    # Now we create a Ferrite grid out of it. Note that we also call toelements
    # with our surface element dimension to obtain these.
    nodes = tonodes()
    elements, _ = toelements(2)
    gmsh.finalize()
Grid(elements, nodes)