#include "pr.hpp"
#include <iostream>
#include <memory>
#include <vector>

#include "kdtree.h"


int main() {
    using namespace pr;

    //Stopwatch sw;

    //PCG rng;
    //std::minstd_rand minrand;
    //for (int i = 0; i < 10000; i++)
    //{
    //    static std::vector<int> data;
    //    data.clear();

    //    int size = 10 + rng.uniform() % 64;
    //    for (int j = 0; j < size; j++)
    //    {
    //        data.push_back(rng.uniform() % 10);
    //    }

    //    for (int j = 0; j < 1000; j++)
    //    {
    //        int i_th = rng.uniform() % size;
    //    
    //        //std::nth_element(data.begin(), data.begin() + i_th, data.end());
    //        kdtree::quick_select(data.data(), data.size(), i_th);

    //        int i_th_val = data[i_th];

    //        std::shuffle(data.begin(), data.end(), minrand);

    //        std::nth_element(data.begin(), data.begin() + i_th, data.end());

    //        PR_ASSERT(data[i_th] == i_th_val);

    //        // 
    //    }
    //}

    //printf("%f\n", sw.elapsed());

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 0, 0, 4 };
    camera.lookat = { 0, 0, 0 };

    double e = GetElapsedTime();

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

        static float radius = 0.2f;
        static glm::vec3 p = { 0.5f, 0.5f, 0 };
        ManipulatePosition(camera, &p, 0.2f);

        enum
        {
            Mode_2D_closest,
            Mode_2D_radius,
            Mode_3D_radius
        };
        static int mode = Mode_3D_radius;

        PCG rng;

        if (mode == Mode_2D_closest || mode == Mode_2D_radius)
        {
            p.z = 0.0f;

            std::vector<kdtree::VecN<2>> points;

            for (int i = 0; i < 128; i++)
            {
                glm::vec2 p = { rng.uniformf(), rng.uniformf() };

                DrawPoint({ p.x, p.y, 0.0f }, { 255, 255, 255 }, 8);
                points.push_back({ p.x, p.y });
            }

            std::vector<kdtree::Node<2>> nodes(kdtree::storage_size(points.size()));
            kdtree::build(nodes.data(), points.data(), points.size());

            kdtree::volume_visit(nodes.data(), points.size(), [](kdtree::VecN<2> lower, kdtree::VecN<2> upper) {
                pr::DrawAABB({ lower[0], lower[1], 0.0f }, { upper[0], upper[1], 0.0f }, { 128, 128, 128 });
            });
            
            if(mode == Mode_2D_closest)
            {
                int idx = kdtree::closest_query_stackfree(nodes.data(), points.size(), { p.x, p.y });
                kdtree::VecN<2> closest = points[idx];

                DrawPoint({ closest[0], closest[1], 0.0f }, { 255, 0, 0 }, 8);
                DrawLine({ closest[0], closest[1], 0.0f }, { p[0], p[1], 0.0f }, { 128, 0, 0 }, 2);
            }
            else
            {
                pr::DrawCircle(p, { 0, 0, 1 }, { 255, 255,0 }, radius);

                kdtree::radius_query(nodes.data(), points.size(), { p.x, p.y }, radius, [&points](int index) {
                    auto p = points[index];
                    DrawPoint({ p[0], p[1], 0.0f }, { 255, 0, 0 }, 8);
                });
            }
        }
        else if (mode == Mode_3D_radius)
        {
            std::vector<kdtree::VecN<3>> points;

            for (int i = 0; i < 128; i++)
            {
                glm::vec3 p = { rng.uniformf(), rng.uniformf(), rng.uniformf() * 0.2f };

                DrawPoint(p, { 255, 255, 255 }, 8);
                points.push_back({ p.x, p.y,  p.z });
            }

            std::vector<kdtree::Node<3>> nodes(kdtree::storage_size(points.size()));
            kdtree::build(nodes.data(), points.data(), points.size());

            kdtree::volume_visit(nodes.data(), points.size(), [](kdtree::VecN<3> lower, kdtree::VecN<3> upper) {
                pr::DrawAABB({ lower[0], lower[1], lower[2] }, { upper[0], upper[1], upper[2] }, { 128, 128, 128 });
            });

            pr::DrawSphere(p, radius, { 255, 255, 0 }, 16, 16);

            kdtree::radius_query(nodes.data(), points.size(), { p.x, p.y, p.z }, radius, [&points](int index) {
                auto p = points[index];
                DrawPoint({ p[0], p[1], p[2] }, { 255, 0, 0 }, 8);
            });
        }

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        ImGui::SliderFloat("radius", &radius, 0, 1);

        ImGui::Text("Demo Mode");
        ImGui::RadioButton("2D closest", &mode, Mode_2D_closest);
        ImGui::RadioButton("2D radius", &mode, Mode_2D_radius);
        ImGui::RadioButton("3D radius", &mode, Mode_3D_radius);

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
