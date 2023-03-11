#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/IO/Nef_polyhedron_iostream_3.h>
#include <CGAL/minkowski_sum_3.h>

typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
typedef CGAL::Nef_polyhedron_3<Kernel>                    Nef_polyhedron;
typedef Kernel::Point_3                                   Point_3;
typedef Point_3*                                          point_iterator;
typedef std::pair<point_iterator,point_iterator>          point_range;
typedef std::list<point_range>                            polyline;

class PolytopeAlgebra_3
{
public:
    PolytopeAlgebra_3()
    {

    }

    PolytopeAlgebra_3 operator + (PolytopeAlgebra_3 const other) const 
    {
        // + in tropical geometry is taking the convex hull of the union
        
    }

    PolytopeAlgebra_3 operator * (PolytopeAlgebra_3 const other) const 
    {
        // * in tropical geometry is taking the minkowski sum
        return CGAL::minkowski_sum_3(N, other.N)
    }
private:
    Nef_polyhedron N;
}
