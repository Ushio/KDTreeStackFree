
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
        float boundary;
    };

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

    void build(std::vector<Node>* nodes, Vec2* point_beg, Vec2* point_end, Vec2 lower, Vec2 upper)
    {
        enum { dims = 2 };
        int axis = large_axis(lower, upper, dims);

        int L = lbalanced(point_end - point_beg);

        printf("");
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
}