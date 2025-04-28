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

        PCG rng;

        std::vector<kdtree::Vec2> points;

        for (int i = 0; i < 100; i++)
        {
            glm::vec2 p = { rng.uniformf(), rng.uniformf() };

            DrawPoint({ p.x, p.y, 0.0f }, { 255, 255, 255 }, 8);
            points.push_back({ p.x, p.y });
        }

        std::vector<kdtree::Node> nodes;
        kdtree::build(&nodes, points.data(), points.size());

        static float radius = 0.2f;
        static glm::vec3 p = { 0.5f, 0.5f, 0 };
        ManipulatePosition(camera, &p, 0.2f);
        p.z = 0.0f;

        pr::DrawCircle(p, { 0, 0, 1 }, { 255, 255,0 }, radius);

        //kdtree::radius_query(nodes.data(), points.size(), { p.x, p.y }, radius, [](kdtree::Vec2 p) {
        //    pr::DrawCircle({ p[0], p[1], 0.0f}, {0, 0, 1}, {255, 0, 0}, 0.01f);
        //});
        {
            kdtree::Vec2 closest = kdtree::closest_query(nodes.data(), points.size(), { p.x, p.y });
            pr::DrawCircle({ closest[0], closest[1], 0.0f }, { 0, 0, 1 }, { 255, 0, 0 }, 0.01f);
        }

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        ImGui::SliderFloat("radius", &radius, 0, 1);

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
