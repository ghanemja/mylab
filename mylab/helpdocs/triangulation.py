"""Help text for triangulation and delaunayTriangulation objects and their methods."""

_TR = "TR is a triangulation or delaunayTriangulation. Also callable as TR.{name}(...)."

HELP = {
    "triangulation": (["TR = triangulation(T, P)", "TR = triangulation(T, x, y)", "TR = triangulation(T, x, y, z)"],
                      "Triangulation object from a connectivity list T (m-by-3 triangles or m-by-4 tetrahedra, "
                      "rows of vertex IDs) and points P (n-by-2 or n-by-3). The read-only properties Points and "
                      "ConnectivityList hold the data; TR(i, j) indexes the connectivity list and size(TR) is its "
                      "size. Methods: edges, freeBoundary, neighbors, vertexAttachments, edgeAttachments, "
                      "faceNormal, vertexNormal, incenter, circumcenter, barycentricToCartesian, "
                      "cartesianToBarycentric, nearestNeighbor, pointLocation, isConnected, featureEdges, size. "
                      "triplot, trisurf and trimesh accept TR directly."),
    "delaunayTriangulation": (["DT = delaunayTriangulation(P)", "DT = delaunayTriangulation(x, y)",
                               "DT = delaunayTriangulation(x, y, z)", "DT = delaunayTriangulation(P, C)",
                               "DT = delaunayTriangulation(x, y, C)", "DT = delaunayTriangulation()"],
                              "Delaunay triangulation of 2-D or 3-D points (counterclockwise triangles in 2-D). "
                              "Duplicate points are removed with a warning. C (k-by-2 point indices) gives 2-D "
                              "edge constraints, which are recovered by edge flips; constraints that cross each "
                              "other are not supported. DT is a triangulation (all its methods work) with the "
                              "extra methods convexHull, voronoiDiagram and isInterior, and the property "
                              "Constraints. Assigning DT.Points or DT.Constraints retriangulates."),
    "edges": ("E = edges(TR)",
              "The edges of the triangulation, as an m-by-2 matrix of vertex IDs with the smaller ID first, unique "
              "and sorted by rows. " + _TR.format(name="edges")),
    "freeBoundary": (["F = freeBoundary(TR)", "[F, P] = freeBoundary(TR)"],
                     "Facets referenced by only one triangle (edges, chained into loops, in 2-D) or tetrahedron "
                     "(triangles with outward normals). With two outputs F indexes the boundary points P instead "
                     "of TR.Points. " + _TR.format(name="freeBoundary")),
    "vertexAttachments": (["C = vertexAttachments(TR)", "C = vertexAttachments(TR, VID)"],
                          "Column cell array: C{i} is a row vector of the IDs of the triangles (tetrahedra) that "
                          "use vertex i (or VID(i)). " + _TR.format(name="vertexAttachments")),
    "edgeAttachments": (["C = edgeAttachments(TR, V1, V2)", "C = edgeAttachments(TR, E)"],
                        "Column cell array of the IDs of the triangles (tetrahedra) attached to each edge. "
                        + _TR.format(name="edgeAttachments")),
    "faceNormal": (["N = faceNormal(TR)", "N = faceNormal(TR, ID)"],
                   "Unit normals of the triangles of a 3-D surface triangulation, one row per triangle. "
                   + _TR.format(name="faceNormal")),
    "vertexNormal": (["N = vertexNormal(TR)", "N = vertexNormal(TR, VID)"],
                     "Unit normals at the vertices of a 3-D surface triangulation: the area-weighted average of "
                     "the normals of the attached triangles. " + _TR.format(name="vertexNormal")),
    "incenter": (["C = incenter(TR)", "C = incenter(TR, ID)", "[C, r] = incenter(TR, ...)"],
                 "Centers (and radii) of the circles or spheres inscribed in the triangles or tetrahedra. "
                 + _TR.format(name="incenter")),
    "circumcenter": (["C = circumcenter(TR)", "C = circumcenter(TR, ID)", "[C, r] = circumcenter(TR, ...)"],
                     "Centers (and radii) of the circumscribed circles or spheres of the triangles or tetrahedra. "
                     + _TR.format(name="circumcenter")),
    "barycentricToCartesian": ("C = barycentricToCartesian(TR, ID, B)",
                               "Cartesian coordinates of points given by barycentric coordinates B (one row per "
                               "ID) relative to the triangles or tetrahedra ID. "
                               + _TR.format(name="barycentricToCartesian")),
    "cartesianToBarycentric": ("B = cartesianToBarycentric(TR, ID, C)",
                               "Barycentric coordinates of the points C (one row per ID) relative to the triangles "
                               "or tetrahedra ID; the rows of B sum to 1. "
                               + _TR.format(name="cartesianToBarycentric")),
    "nearestNeighbor": (["ID = nearestNeighbor(TR, QP)", "ID = nearestNeighbor(TR, x, y)",
                         "ID = nearestNeighbor(TR, x, y, z)", "[ID, d] = nearestNeighbor(TR, ...)"],
                        "IDs of the triangulation vertices closest to the query points, and the distances. "
                        + _TR.format(name="nearestNeighbor")),
    "pointLocation": (["ID = pointLocation(TR, QP)", "ID = pointLocation(TR, x, y)",
                       "ID = pointLocation(TR, x, y, z)", "[ID, B] = pointLocation(TR, ...)"],
                      "The triangle (tetrahedron) that contains each query point, NaN outside the triangulation, "
                      "and the barycentric coordinates in it. Not for 3-D surface triangulations. "
                      + _TR.format(name="pointLocation")),
    "isConnected": (["tf = isConnected(TR, V1, V2)", "tf = isConnected(TR, E)"],
                    "True where the two vertices are joined by an edge of the triangulation (column vector). "
                    + _TR.format(name="isConnected")),
    "featureEdges": ("E = featureEdges(TR, filterangle)",
                     "Sharp edges of a triangle mesh: edges whose two triangles' normals differ by more than "
                     "filterangle (radians), plus edges shared by one triangle or by more than two. "
                     + _TR.format(name="featureEdges")),
    "convexHull": (["K = convexHull(DT)", "[K, AV] = convexHull(DT)"],
                   "Convex hull of a delaunayTriangulation: in 2-D a closed counterclockwise list of vertex IDs, "
                   "in 3-D a triangle list with outward normals; AV is the area or volume. Also DT.convexHull()."),
    "voronoiDiagram": ("[V, r] = voronoiDiagram(DT)",
                       "Voronoi diagram of an unconstrained delaunayTriangulation. V holds the Voronoi vertices "
                       "(the circumcenters of the triangles, after a first row of Inf for the vertex at infinity); "
                       "r{i} lists the indices into V of the region of point i, in adjacent order in 2-D, starting "
                       "with 1 for unbounded regions. Also DT.voronoiDiagram()."),
    "isInterior": ("TF = isInterior(DT)",
                   "For a 2-D delaunayTriangulation with edge constraints: true for the triangles inside the region "
                   "bounded by the constraints (inside and outside alternate across nested boundaries). Column "
                   "vector, one element per triangle. Also DT.isInterior()."),
}
