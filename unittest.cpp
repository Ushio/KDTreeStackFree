#include "catch_amalgamated.hpp"
#include "pr.hpp"
#include "kdtree.h"

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

TEST_CASE("tree", "") {
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