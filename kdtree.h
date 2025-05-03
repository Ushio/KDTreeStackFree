#pragma once
#include <stdint.h>
#include <intrin.h>

namespace kdtree
{
    template <class T>
    inline T ss_max(T x, T y)
    {
        return (x < y) ? y : x;
    }

    template <class T>
    inline T ss_min(T x, T y)
    {
        return (y < x) ? y : x;
    }
    template <class T>
    inline T ss_clamp(T x, T lower, T upper)
    {
        return ss_min(ss_max(x, lower), upper);
    }

    template <int N>
    struct VecN
    {
        float xs[N];
        float& operator[](int i) { return xs[i]; }
        const float& operator[](int i) const { return xs[i]; }
    };

    template <class T>
    inline float distanceSquared(T a, T b, int dims)
    {
        float d2 = 0.0f;
        for (int i = 0; i < dims; i++)
        {
            float d = a[i] - b[i];
            d2 += d * d;
        }
        return d2;
    }

    inline int clz(uint32_t x)
    {
#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
        unsigned long scan;
        if (_BitScanReverse(&scan, x) == 0)
        {
            return 32;
        }
        return 31 - scan;
#else
        return __clz(x);
#endif
    }

    inline uint32_t depth( uint32_t nNodes )
    {
        return 32 - clz(nNodes);
    }
    inline uint32_t node_capacity(uint32_t depth)
    {
        return (1u << depth) - 1;
    }

    // return number of node on the left
    inline uint32_t lbalanced(uint32_t nNodes)
    {
        int d = depth(nNodes);
        int lMax = node_capacity(d - 1); // maximum number of left tree
        int rMin = lMax / 2;             // minimum number of right tree
        return ss_min((int)nNodes - 1 - rMin, lMax); // put maximum on the left
    }

    template <class T, class Comp>
    int partition_by_mid(T* data, int beg, int end, Comp comp)
    {
        if (beg + 1 == end)
        {
            return beg;
        }
        int pivot = (beg + end) / 2;
        std::swap(data[pivot], data[end - 1]);

        T pivot_value = data[end - 1];
        int L_index = beg;
        for (int i = beg; i < end - 1; i++)
        {
            if (comp(data[i], pivot_value))
            {
                std::swap(data[L_index], data[i]);
                L_index++;
            }
        }
        std::swap(data[L_index], data[end - 1]);
        return L_index;
    }

    template <class T, class Comp>
    void quick_select(T* data, int n, int i, Comp comp)
    {
        int beg = 0;
        int end = n;
        for (;;)
        {
            int pivot_to = partition_by_mid(data, beg, end, comp);
            if (pivot_to == i)
            {
                break;
            }
            else if (i < pivot_to)
            {
                end = pivot_to;
            }
            else
            {
                beg = pivot_to + 1;
            }
        }
    }
    template <class T>
    void quick_select(T* data, int n, int i)
    {
        quick_select(data, n, i, [](T a, T b) { return a < b; });
    }


    template <class T>
    int large_axis(T lower, T upper, int dims)
    {
        float wide = -1.0f;
        int axis = -1;
        for (int i = 0; i < dims; i++)
        {
            float thisWide = upper[i] - lower[i];
            if (wide < thisWide)
            {
                axis = i;
                wide = thisWide;
            }
        }
        return axis;
    }

    // 1 based array
    inline int l_child(int node)
    {
        return node * 2;
    }
    inline int r_child(int node)
    {
        return node * 2 + 1;
    }
    inline int parent(int node)
    {
        return node / 2;
    }

    template <int dims>
    struct Node
    {
        VecN<dims> p;
        int src_index;
        int axis;
    };

    inline uint32_t storage_size(uint32_t nPoints)
    {
        int capacity = node_capacity(depth(nPoints));
        return capacity + 1;
    }

    namespace details
    {
        template <int dims>
        inline void build(Node<dims>* nodes, int node_idx, const VecN<dims>* ps, int* indices, int point_beg, int point_end, VecN<dims> lower, VecN<dims> upper)
        {
            if (point_beg == point_end)
            {
                return;
            }

            int axis = large_axis(lower, upper, dims);

            int n = point_end - point_beg;
            int L = lbalanced(n);
#if 1
            quick_select(indices + point_beg, n, L, [axis, ps](int a, int b) { return ps[a][axis] < ps[b][axis]; });
#else
            std::nth_element(indices + point_beg, indices + point_beg + L, indices + point_end, [axis, ps](int a, int b) { return ps[a][axis] < ps[b][axis]; });
#endif

            int mid_index = indices[point_beg + L];
            Node<dims> node;
            node.axis = axis;
            node.p = ps[mid_index];
            node.src_index = mid_index;
            nodes[node_idx] = node;

            VecN<dims> lUpper = upper;
            lUpper[axis] = node.p[node.axis];

            VecN<dims> rLower = lower;
            rLower[axis] = node.p[node.axis];

            build(nodes, l_child(node_idx), ps, indices, point_beg, point_beg + L, lower, lUpper);
            build(nodes, r_child(node_idx), ps, indices, point_beg + L + 1, point_end, rLower, upper );
        }
    }
    template <int dims>
    void build(Node<dims>* nodes, const VecN<dims>* ps, int nPoints)
    {
        VecN<dims> lower;
        VecN<dims> upper;
        for (int i = 0; i < dims; i++)
        {
            lower[i] = +FLT_MAX;
            upper[i] = -FLT_MAX;
        }
        for (int j = 0 ; j < nPoints; j++)
        {
            for (int i = 0; i < dims; i++)
            {
                lower[i] = ss_min(lower[i], ps[j][i]);
                upper[i] = ss_max(upper[i], ps[j][i]);
            }
        }

        std::vector<int> indices(nPoints);
        for (int i = 0; i < nPoints; i++)
        {
            indices[i] = i;
        }
        details::build(nodes, 1, ps, indices.data(), 0, nPoints, lower, upper);
    }

    namespace details {
        template<int dims, class F>
        void volume_visit(const Node<dims>* nodes, int nPoints, int node_idx, VecN<dims> lower, VecN<dims> upper, F f)
        {
            if (nPoints < node_idx)
            {
                f(lower, upper);
                return;
            }

            auto node = nodes[node_idx];

            VecN<dims> lUpper = upper;
            lUpper[node.axis] = node.p[node.axis];

            VecN<dims> rLower = lower;
            rLower[node.axis] = node.p[node.axis];

            volume_visit(nodes, nPoints, l_child(node_idx), lower, lUpper, f);
            volume_visit(nodes, nPoints, r_child(node_idx), rLower, upper, f);
        }
    }

    template<int dims, class F>
    void volume_visit(const Node<dims>* nodes, int nPoints, F f)
    {
        VecN<dims> lower;
        VecN<dims> upper;
        for (int i = 0; i < dims; i++)
        {
            lower[i] = +FLT_MAX;
            upper[i] = -FLT_MAX;
        }

        for (int j = 1; j < nPoints + 1; j++)
        {
            for (int i = 0; i < dims; i++)
            {
                lower[i] = ss_min(lower[i], nodes[j].p[i]);
                upper[i] = ss_max(upper[i], nodes[j].p[i]);
            }
        }

        details::volume_visit(nodes, nPoints, 1, lower, upper, f);
    }


    template <int dims>
    inline int closest_query(const Node<dims>* nodes, int nPoints, VecN<dims> point)
    {
        int index = 0;

        float r2 = FLT_MAX;
        int closest_index = -1;

        int curr_node = 1;
        int prev_node = -1;
        while (0 < curr_node)
        {
            int parent_node = parent(curr_node);

            if( nPoints < curr_node ) // done. so go back
            {
                prev_node = curr_node;
                curr_node = parent_node;
                continue;
            }

            bool descent = prev_node < curr_node;

            auto node = nodes[curr_node];

            if (descent)
            {
                float d2 = distanceSquared(node.p, point, dims);
                if (d2 < r2)
                {
                    r2 = d2;
                    closest_index = node.src_index;
                }
                // pr::DrawCircle({ node.p[0], node.p[1], 0.0f }, { 0, 0, 1 }, { 0, 0, 255 }, 0.02f);
            }

            int near_node = l_child(curr_node);
            int far_node = r_child(curr_node);

            float d = point[node.axis] - node.p[node.axis];
            if (0.0f < d)
            {
                std::swap(near_node, far_node);
            }

            int next_node;
            if (descent)
            {
                next_node = near_node;
            }
            else if( prev_node == near_node ) // meaning that far_node may need to traverse
            {
                bool traverse_far = d * d < r2;
                next_node = traverse_far ? far_node : parent_node;
            }
            else // meaning that both have done
            {
                next_node = parent_node;
            }

            prev_node = curr_node;
            curr_node = next_node;
        }
        return closest_index;
    }

    template <int dims, class F>
    void radius_query(const Node<dims>* nodes, int nPoints, VecN<dims> point, float radius, F f)
    {
        float r2 = radius * radius;

        int curr_node = 1;
        int prev_node = -1;

        while (0 < curr_node)
        {
            int parent_node = parent(curr_node);

            if (nPoints < curr_node) // done. so go back
            {
                prev_node = curr_node;
                curr_node = parent_node;
                continue;
            }

            bool descent = prev_node < curr_node;

            auto node = nodes[curr_node];

            if (descent)
            {
                //pr::DrawCircle({ node.p[0], node.p[1], 0.0f }, { 0, 0, 1 }, { 0, 0, 255 }, 0.02f);

                float d2 = distanceSquared(node.p, point, dims);
                if (d2 < r2)
                {
                    f(node.src_index);
                }
            }

            int near_node = l_child(curr_node);
            int far_node = r_child(curr_node);

            float d = point[node.axis] - node.p[node.axis];
            if (0.0f < d)
            {
                std::swap(near_node, far_node);
            }

            int next_node;
            if (descent)
            {
                next_node = near_node;
            }
            else if (prev_node == near_node) // meaning that far_node may need to traverse
            {
                bool traverse_far = d * d < r2;
                next_node = traverse_far ? far_node : parent_node;
            }
            else // meaning that both have done
            {
                next_node = parent_node;
            }

            prev_node = curr_node;
            curr_node = next_node;
        }
    }

    namespace details
    {
        template <int dims, class F>
        void radius_query_stackful(const Node<dims>* nodes, int nPoints, int curr_node, const VecN<dims>& point, float r2, F f)
        {
            if (nPoints < curr_node)
            {
                return;
            }

            auto node = nodes[curr_node];

            //pr::DrawCircle({ node.p[0], node.p[1], 0.0f }, { 0, 0, 1 }, { 0, 0, 255 }, 0.02f);

            float d2 = distanceSquared(node.p, point, dims);
            if (d2 < r2)
            {
                f(node.src_index);
            }

            int near_node = l_child(curr_node);
            int far_node = r_child(curr_node);

            float d = point[node.axis] - node.p[node.axis];
            if (0.0f < d)
            {
                std::swap(near_node, far_node);
            }

            radius_query_stackful(nodes, nPoints, near_node, point, r2, f);
            bool traverse_far = d * d < r2;
            if (traverse_far)
            {
                radius_query_stackful(nodes, nPoints, far_node, point, r2, f);
            }
        }
    }

    template <int dims, class F>
    void radius_query_stackful(const Node<dims>* nodes, int nPoints, const VecN<dims>& point, float radius, F f)
    {
        details::radius_query_stackful(nodes, nPoints, 1, point, radius * radius, f);
    }
}