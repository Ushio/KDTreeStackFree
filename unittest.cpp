#include "catch_amalgamated.hpp"
#include "pr.hpp"
#include "kdtree.h"
#include "nanoflann.hpp"

TEST_CASE("int", "") {
    using namespace pr;

    PCG rng;
    std::minstd_rand minrand;
    for (int i = 0; i < 10000; i++)
    {
        static std::vector<int> data;
        data.clear();

        int size = 1 + rng.uniform() % 64;
        for (int j = 0; j < size; j++)
        {
            data.push_back(rng.uniform() % 10);
        }

        for (int j = 0; j < 1000; j++)
        {
            int i_th = rng.uniform() % size;

            kdtree::quick_select(data.data(), data.size(), i_th);

            auto i_th_val = data[i_th];

            std::shuffle(data.begin(), data.end(), minrand);

            std::nth_element(data.begin(), data.begin() + i_th, data.end());

            REQUIRE(data[i_th] == i_th_val);
        }
    }
}

TEST_CASE("float", "") {
    using namespace pr;

    PCG rng;
    std::minstd_rand minrand;
    for (int i = 0; i < 10000; i++)
    {
        static std::vector<glm::vec3> data;
        data.clear();

        int size = 1 + rng.uniform() % 64;
        for (int j = 0; j < size; j++)
        {
            data.push_back({ rng.uniformf(), rng.uniformf(), rng.uniformf() });
        }

        for (int j = 0; j < 1000; j++)
        {
            int i_th = rng.uniform() % size;
            int dim = rng.uniform() % 3;

            kdtree::quick_select(data.data(), data.size(), i_th, [dim](glm::vec3 a, glm::vec3 b) { return  a[dim] < b[dim]; });
            //std::nth_element(data.begin(), data.begin() + i_th, data.end(), [dim](glm::vec3 a, glm::vec3 b) { return a[dim] < b[dim]; });

            auto i_th_val = data[i_th];

            std::shuffle(data.begin(), data.end(), minrand);

            std::nth_element(data.begin(), data.begin() + i_th, data.end(), [dim](glm::vec3 a, glm::vec3 b) { return  a[dim] < b[dim]; });

            REQUIRE(data[i_th][dim] == i_th_val[dim]);
        }
    }
}

TEST_CASE("lbalanced", "") {
    REQUIRE(kdtree::lbalanced(1) == 0);
    REQUIRE(kdtree::lbalanced(2) == 1);
    REQUIRE(kdtree::lbalanced(3) == 1);
    REQUIRE(kdtree::lbalanced(4) == 2);
    REQUIRE(kdtree::lbalanced(5) == 3);
    REQUIRE(kdtree::lbalanced(6) == 3);
    REQUIRE(kdtree::lbalanced(7) == 3);
    REQUIRE(kdtree::lbalanced(8) == 4);
    REQUIRE(kdtree::lbalanced(9) == 5);

    //for (int n = 1; n < 32; n++)
    //{
    //    int d = kdtree::depth(n);

    //    int rMin = kdtree::node_capacity(kdtree::ss_max(d - 2, 0));  // minimum number of right tree
    //    int lMax = kdtree::node_capacity(d - 1);             // maximum number of left tree

    //    int L = kdtree::lbalanced(n);
    //    int R = n - 1 - L;
    //    printf("n=%d, depth=%d, lMax=%d, rMin=%d, %d-%d\n", n, d, lMax, rMin, L, R);
    //}
}

template<int dims>
struct PointCloud
{
    std::vector<kdtree::VecN<dims>> points;

    inline size_t kdtree_get_point_count() const { return points.size(); }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return points[idx][dim];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};

TEST_CASE("tree", "") {
    using namespace pr;

    PCG rng;

    enum {
        dims = 3
    };
    for (int i = 0; i < 1000; i++)
    {
        PointCloud<dims> cloud;

        for (int j = 0; j < 1000; j++)
        {
            kdtree::VecN<dims> p;
            for (int k = 0; k < dims; k++)
            {
                p[k] = rng.uniformf();
            }
            cloud.points.push_back(p);
        }

        // nanoflann
        using kdAdapter = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, PointCloud<dims>>,
            PointCloud<dims>, dims /* dim */
        >;

        kdAdapter kdtree(dims /*dim*/, cloud, { 10 /* max leaf */ });

        // kdtree
        std::vector<kdtree::Node<dims>> nodes(kdtree::storage_size(cloud.points.size()));
        kdtree::build(nodes.data(), cloud.points.data(), cloud.points.size());

        for (int j = 0; j < 1000; j++)
        {
            float radius = rng.uniformf() * 0.2f;
            kdtree::VecN<dims> query_pt;
            for (int k = 0; k < dims; k++)
            {
                query_pt[k] = rng.uniformf();
            }

            uint32_t ref_hash = 0;
            {
                static std::vector<nanoflann::ResultItem<uint32_t, float>> ret_matches;
           
                size_t nMatches = kdtree.radiusSearch(query_pt.xs, radius * radius, ret_matches);

                for (size_t i = 0; i < nMatches; i++)
                {
                    uint32_t index = ret_matches[i].first;
                    ref_hash ^= index;
                }
            }

            uint32_t my_hash = 0;
            {
                kdtree::radius_query(nodes.data(), cloud.points.size(), query_pt, radius, [&my_hash](uint32_t index) {
                    my_hash ^= index;
                });
            }

            REQUIRE(my_hash == ref_hash);
        }
    }
}