
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


    struct Vec2
    {
        float xs[2];
        float& operator[](int i) { return xs[i]; }
        const float& operator[](int i) const { return xs[i]; }
    };


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

    struct Node
    {
        Vec2 p;
        int axis;
    };

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

    inline void build(Node* nodes, int node_idx, Vec2* ps, int point_beg, int point_end, Vec2 lower, Vec2 upper)
    {
        if (point_beg == point_end)
        {
            return;
        }

        enum { dims = 2 };
        int axis = large_axis(lower, upper, dims);

        int n = point_end - point_beg;
        int L = lbalanced(n);
        quick_select(ps + point_beg, n, L, [axis](Vec2 a, Vec2 b) { return a[axis] < b[axis]; });

        //for (int i = 0; i < L; i++)
        //{
        //    auto p = ps[i];
        //    pr::DrawCircle({ p[0], p[1], 0.0f}, {0, 0, 1}, {255, 255,0}, 0.01f);
        //}
        //for (int i = L + 1; i < n; i++)
        //{
        //    auto p = ps[i];
        //    pr::DrawCircle({ p[0], p[1], 0.0f }, { 0, 0, 1 }, { 255,0 ,255 }, 0.01f);
        //}

        Node node;
        node.axis = axis;
        node.p = ps[point_beg + L];
        nodes[node_idx] = node;

        {
            Vec2 lineBeg = lower;
            Vec2 lineEnd = upper;
            lineBeg[axis] = node.p[node.axis];
            lineEnd[axis] = node.p[node.axis];
            pr::DrawLine({ lineBeg[0],  lineBeg[1], 0.0f }, { lineEnd[0],  lineEnd[1], 0.0f }, { 128, 128, 128 });
        }

        Vec2 lUpper = upper;
        lUpper[axis] = node.p[node.axis];

        Vec2 rLower = lower;
        rLower[axis] = node.p[node.axis];

        build(nodes, l_child(node_idx), ps, point_beg, point_beg + L, lower, lUpper);
        build(nodes, r_child(node_idx), ps, point_beg + L + 1, point_end, rLower, upper );
    }
    void build(std::vector<Node>* nodes, Vec2* ps, int nPoints)
    {
        enum { dims = 2 };
        Vec2 lower;
        Vec2 upper;
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

        pr::DrawAABB({ lower[0], lower[1], 0.0f }, { upper[0], upper[1], 0.0f }, { 255, 255, 255 });

        int capacity = node_capacity(depth(nPoints));
        nodes->resize(capacity + 1 /*1 based array*/);
        build(nodes->data(), 1, ps, 0, nPoints, lower, upper);
    }
}