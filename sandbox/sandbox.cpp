#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/IO/Nef_polyhedron_iostream_3.h>
#include <CGAL/minkowski_sum_3.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/Surface_mesh.h>

typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
typedef CGAL::Nef_polyhedron_3<Kernel>                    Nef_polyhedron;
typedef Kernel::Point_3                                   Point_3;
typedef Point_3*                                          point_iterator;
typedef std::pair<point_iterator,point_iterator>          point_range;
typedef std::list<point_range>                            polyline;
typedef CGAL::Surface_mesh<Point_3>						  PolygonMesh;

/*
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
		PolytopeAlgebra_3 res = PolytopeAlgebra_3();
		res.N = CGAL::minkowski_sum_3(N, other.N);
		return res;
    }
	
	Nef_polyhedron N;
};
//*/

int main()
{
	Point_3 pl1[3] =
    {
		Point_3(-100,0,0),
		Point_3(40,-70,0),
		Point_3(40,50,40)
  	};
	Point_3 pl2[3] =
    {
		Point_3(-90,-60,60),
		Point_3(0, 0, -100),
		Point_3(30,0,150)
  	};
	Nef_polyhedron N1(pl1, pl1+3);
	Nef_polyhedron N2(pl2, pl2+3);
	PolygonMesh res;
	CGAL::extreme_points_3(N1 + N2, res, Kernel);
}