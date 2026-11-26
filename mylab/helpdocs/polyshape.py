"""Help text for polyshape objects, their methods and nsidedpoly."""

_P = "p is a polyshape. Also callable as p.{name}(...)."

HELP = {
    "polyshape": (["p = polyshape(x, y)", "p = polyshape(P)", "p = polyshape({x1, x2}, {y1, y2})", "p = polyshape()",
                   "p = polyshape(..., 'Simplify', tf, 'KeepCollinearPoints', tf)",
                   "p = polyshape(..., 'SolidBoundaryOrientation', 'auto' | 'cw' | 'ccw')"],
                  "A 2-D polygon that may have several regions and holes. Vertices come as x and y vectors, an "
                  "N-by-2 matrix or cell arrays with one boundary per cell; NaN rows separate boundaries. By default "
                  "the polygon is simplified: repeated vertices, crossings and overlapping boundaries are repaired "
                  "(with the warning MATLAB:polyshape:repairedBySimplify), collinear vertices are dropped unless "
                  "KeepCollinearPoints is true, and solid boundaries are stored clockwise with holes "
                  "counterclockwise. A point is filled when an odd number of the given boundaries enclose it "
                  "('auto'), or by orientation with 'cw' / 'ccw'. Properties: Vertices (boundaries separated by NaN "
                  "rows), NumRegions, NumHoles. Polyshapes form arrays: [p1 p2], p(k), p(k) = q. The methods "
                  "area, union, intersect, xor, convhull, simplify, triangulation, plot and isequal also accept "
                  "polyshapes; intersect(p, L) with an N-by-2 polyline returns the parts inside and outside."),
    "nsidedpoly": (["p = nsidedpoly(n)", "p = nsidedpoly(n, 'Center', [cx cy])",
                    "p = nsidedpoly(n, 'SideLength', s)", "p = nsidedpoly(n, 'Radius', r)"],
                   "A regular polygon with n sides as a polyshape, with one side horizontal at the bottom. The "
                   "size is the circumscribed radius (1 by default) or the side length; the center defaults to "
                   "the origin."),
    "perimeter": (["L = perimeter(p)", "L = perimeter(p, I)"],
                  "Total length of all boundaries of p (holes included), or of the boundaries I. For an array of "
                  "polyshapes, one value per element. " + _P.format(name="perimeter")),
    "centroid": (["[x, y] = centroid(p)", "[x, y] = centroid(p, I)"],
                 "Center of mass of the filled area of p (holes subtracted), or of each boundary I taken as a "
                 "solid. NaN for an empty polyshape. " + _P.format(name="centroid")),
    "boundingbox": (["[xlim, ylim] = boundingbox(p)", "[xlim, ylim] = boundingbox(p, I)"],
                    "Smallest and largest x and y of the vertices of p (of all elements of an array), or of "
                    "the boundaries I. " + _P.format(name="boundingbox")),
    "isinterior": (["TFin = isinterior(p, x, y)", "TFin = isinterior(p, [x y])", "[TFin, TFon] = isinterior(...)"],
                   "True for the query points that are in the filled area of p or on its boundary (column "
                   "vector); TFon marks the points on a boundary. " + _P.format(name="isinterior")),
    "numboundaries": ("n = numboundaries(p)",
                      "Number of boundaries (solid ones and holes) of p. " + _P.format(name="numboundaries")),
    "numsides": (["n = numsides(p)", "n = numsides(p, I)"],
                 "Number of edges of p, or of each boundary I. " + _P.format(name="numsides")),
    "ishole": (["tf = ishole(p)", "tf = ishole(p, I)"],
               "Column vector, true for the boundaries that are holes. " + _P.format(name="ishole")),
    "issimplified": ("tf = issimplified(p)",
                     "True when p is well defined: no crossing or repeated boundaries and the orientation "
                     "convention (solid clockwise, holes counterclockwise) holds. " + _P.format(name="issimplified")),
    "nearestvertex": (["vid = nearestvertex(p, x, y)", "vid = nearestvertex(p, [x y])",
                       "[vid, bid, ind] = nearestvertex(...)"],
                      "For each query point, the row of p.Vertices of the closest vertex, the boundary it is on "
                      "and its position in that boundary. " + _P.format(name="nearestvertex")),
    "overlaps": (["tf = overlaps(p1, p2)", "TF = overlaps(P)"],
                 "True when two polyshapes share area or touch. For an array P, the symmetric matrix of all "
                 "pairs. " + _P.format(name="overlaps")),
    "boundary": (["[x, y] = boundary(p)", "[x, y] = boundary(p, I)"],
                 "Coordinates of the boundaries of p (or of the boundaries I) as column vectors, each closed by "
                 "repeating its first vertex and separated by NaN. " + _P.format(name="boundary")),
    "regions": ("R = regions(p)",
                "The regions of p as a column of polyshapes, each a solid boundary with the holes inside it. "
                + _P.format(name="regions")),
    "holes": ("H = holes(p)",
              "The holes of p as a column of solid polyshapes. " + _P.format(name="holes")),
    "rmholes": ("q = rmholes(p)", "p without its holes. " + _P.format(name="rmholes")),
    "addboundary": (["q = addboundary(p, x, y)", "q = addboundary(p, P)", "q = addboundary(p, {x1, x2}, {y1, y2})",
                     "q = addboundary(..., 'Simplify', tf, 'KeepCollinearPoints', tf)"],
                    "p with more boundaries; the result is simplified again unless Simplify is false, so a "
                    "boundary inside a region becomes a hole. " + _P.format(name="addboundary")),
    "rmboundary": (["q = rmboundary(p, I)", "q = rmboundary(p, I, 'Simplify', tf)"],
                   "p without the boundaries I. " + _P.format(name="rmboundary")),
    "rmslivers": ("q = rmslivers(p, tol)",
                  "Removes thin spikes and slits: vertices closer than tol to the line through their neighbours "
                  "are dropped, and boundaries narrower than tol vanish. " + _P.format(name="rmslivers")),
    "sortboundaries": (["q = sortboundaries(p, criterion, direction)", "[q, I] = sortboundaries(...)",
                        "q = sortboundaries(p, 'centroid', direction, 'ReferencePoint', [x y])"],
                       "Reorders the boundaries by 'area', 'perimeter', 'numsides' or 'centroid' (distance of "
                       "the boundary centroid from the reference point, the origin by default), 'ascend' or "
                       "'descend'. I gives the old positions. " + _P.format(name="sortboundaries")),
    "translate": (["q = translate(p, [dx dy])", "q = translate(p, dx, dy)"],
                  "p shifted by (dx, dy). " + _P.format(name="translate")),
    "rotate": (["q = rotate(p, theta)", "q = rotate(p, theta, [x y])"],
               "p turned counterclockwise by theta degrees about the origin or the given point. "
               + _P.format(name="rotate")),
    "scale": (["q = scale(p, s)", "q = scale(p, [sx sy])", "q = scale(p, s, [x y])"],
              "p stretched by the factor s (or sx and sy) about the origin or the given point. A negative "
              "factor mirrors the shape. " + _P.format(name="scale")),
    "subtract": (["q = subtract(p1, p2)", "[q, shapeID, vertexID] = subtract(p1, p2)",
                  "q = subtract(..., 'KeepCollinearPoints', tf)"],
                 "The part of p1 outside p2. shapeID tells for each row of q.Vertices which input it comes from "
                 "(0 for a new intersection point) and vertexID its row there. Also union, intersect and xor "
                 "for polyshapes. " + _P.format(name="subtract")),
    "polybuffer": (["q = polybuffer(p, d)", "q = polybuffer(P, 'lines', d)", "q = polybuffer(P, 'points', d)",
                    "q = polybuffer(..., 'JointType', 'round' | 'square' | 'miter', 'MiterLimit', m)"],
                   "Grows p by the distance d (shrinks it for negative d), or makes the region within d of a "
                   "polyline or of points. Round joints follow circular arcs drawn with 64 points per circle. "
                   + _P.format(name="polybuffer")),
    "turningdist": ("d = turningdist(p1, p2)",
                    "Distance between the turning functions (tangent angle against normalized arc length) of "
                    "two single-boundary polyshapes, minimized over the start point and rotation: 0 for "
                    "similar shapes. For arrays, a matrix of all pairs. " + _P.format(name="turningdist")),
}
