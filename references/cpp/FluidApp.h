#pragma once

#include "PBSSubApp.h"
#include "ImGuiHelpers.h"
#include "VecMatDef.h"
#include "CameraHelper.h"
#include <iostream>

#include "helper/Grid2.h"
#include "helper/MACGrid2.h"

#define CAMERA_PAN_SENSITIVITY 0.003f
#define CAMERA_ROTATE_SENSITIVITY 0.003f
#define CAMERA_ZOOM_SENSITIVITY 0.02f

class FluidApp : public PBSSubApp
{
public:
    bool m_simulating = false;
    bool m_single_step = false;

    int m_field_visualized = 0;
    bool m_show_velocity = false;
    bool m_show_grid = false;

    double m_dt = 0.07;
    double m_tolerance = 1e-5;
    double m_max_iter = 1000;
    bool m_use_wind = false;
    bool m_use_maccormack = false;

    int m_res_x = 128;
    int m_res_y = 192;
    double m_size_x = 1.0;
    double m_size_y;
    double m_dx;

    Grid2 p_density;
    Grid2 p_pressure;
    Grid2 p_divergence;
    Grid2 p_vorticity;

    MACGrid2 p_velocity;
    MACGrid2 p_force;

    /// Rendering
    MatrixXF render_V;
    MatrixXI render_F;
    double render_v_scale = 0.1;

    /// Camera state
    PBSCamera app_camera;

    /// Implementation correctness checking
    std::string m_check_info;

    void mainLoop() override
    {
        if (m_simulating || m_single_step)
        {
            m_single_step = false;
            step();
        }
    }

    void resetSimulation()
    {
        // Allocate grids.
        m_dx = m_size_x / m_res_x;
        m_size_y = m_dx * m_res_y;

        p_density = Grid2(m_res_x, m_res_y, m_dx);
        p_pressure = Grid2(m_res_x, m_res_y, m_dx);
        p_divergence = Grid2(m_res_x, m_res_y, m_dx);
        p_vorticity = Grid2(m_res_x, m_res_y, m_dx);

        p_velocity = MACGrid2(m_res_x, m_res_y, m_dx);
        p_force = MACGrid2(m_res_x, m_res_y, m_dx);

        // Get mesh for rendering from any grid.
        p_density.getMesh(render_V, render_F);

        // Initialize fields to zero.
        p_density.reset();
        p_pressure.reset();
        p_divergence.reset();
        p_vorticity.reset();
        p_velocity.reset();
        p_force.reset();

        // Initialize density
        p_density.applySource(0.45, 0.55, 0.1, 0.15);
    }

    void makeAnalysisWindow() override
    {
        if (ImGui::Button("Check Assignment"))
        {
            m_check_info = m_check();
        }
        ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(10, 10, 10, 255));
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 255));
        ImGui::BeginChild("ScrollingRegion", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
        ImGui::TextWrapped("%s", m_check_info.c_str());
        ImGui::EndChild();
        ImGui::PopStyleColor(2);
    }

    void makeConfigWindow() override
    {
        if (ImGui::CollapsingHeader("Simulation Control", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Simulate", &m_simulating);
            if (ImGui::Button("Single Step"))
            {
                m_single_step = true;
            }
            if (ImGui::Button("Reset Simulation"))
            {
                resetSimulation();
            }
        }
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            std::vector<std::string> field_names = {"Density", "Pressure", "Divergence", "Vorticity"};
            ImGui::Combo("Fields", &m_field_visualized, field_names);
            ImGui::Checkbox("Show Velocity", &m_show_velocity);
            ImGui::Checkbox("Show Grid", &m_show_grid);
        }
        if (ImGui::CollapsingHeader("Simulation Parameters", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputDouble("Time Step", &m_dt);
            ImGui::InputDouble("Max Iterations", &m_max_iter);
            ImGui::InputDouble("Tolerance", &m_tolerance);
            ImGui::Checkbox("Use Wind", &m_use_wind);
            ImGui::Checkbox("Use MacCormack Advection", &m_use_maccormack);
        }
    }

    FluidApp()
    {
        /// Initialize camera.
        app_camera.eye = Vector3F(0.5, 0.75, 5);
        app_camera.center = Vector3F(0.5, 0.75, 0);
        app_camera.set_up_direction(Vector3F(0, 1, 0));
        app_camera.height = 0.80;

        /// Initialize simulation state.
        resetSimulation();
    }

    void getViewerData(std::vector<PBSViewerData> &viewer_data, PBSCamera &viewer_camera) override
    {
        viewer_data.clear();

        /// Simulation grid.
        viewer_data.emplace_back();
        PBSViewerData &viewer_data_grid = viewer_data.back();

        Grid2 grid;
        switch (m_field_visualized)
        {
        case 0:
            grid = p_density;
            break;
        case 1:
            grid = p_pressure;
            break;
        case 2:
            grid = p_divergence;
            break;
        case 3:
            grid = p_vorticity;
            break;
        default:
            assert(0);
            break;
        }

        viewer_data_grid.mesh_v = render_V;
        viewer_data_grid.mesh_f = render_F;
        grid.getColors(viewer_data_grid.mesh_c, m_field_visualized == 0 ? false : true, true);
        viewer_data_grid.mesh_flat_material = true;
        viewer_data_grid.show_lines = m_show_grid;

        if (m_show_velocity)
        {
            viewer_data_grid.edge_width = m_dx * 0.1;

            viewer_data_grid.lines_v.resize(m_res_x * m_res_y * 2, 3);
            viewer_data_grid.lines_e.resize(m_res_x * m_res_y, 2);
            viewer_data_grid.lines_c.resize(m_res_x * m_res_y, 3);

            auto &vx = p_velocity.x();
            auto &vy = p_velocity.y();

            for (int y = 0; y < m_res_y; y++)
            {
                for (int x = 0; x < m_res_x; x++)
                {
                    Eigen::RowVector3d p0 = Eigen::RowVector3d((x + 0.5) * m_dx, (y + 0.5) * m_dx, 0);
                    Eigen::RowVector3d v = Eigen::RowVector3d((vx(x, y) + vx(x + 1, y)) / 2.0, (vy(x, y) + vy(x, y + 1)) / 2.0, 0);
                    Eigen::RowVector3d p1 = p0 + v * render_v_scale;

                    int i = (y * m_res_x + x);
                    viewer_data_grid.lines_v.row(2 * i) = p0;
                    viewer_data_grid.lines_v.row(2 * i + 1) = p1;
                    viewer_data_grid.lines_e.row(i) = Eigen::RowVector2i(2 * i, 2 * i + 1);
                }
            }
            viewer_data_grid.lines_c.setZero();
            if (m_field_visualized == 0)
            {
                viewer_data_grid.lines_c.col(1).setConstant(1.0);
            }
            else
            {
                viewer_data_grid.lines_c.setOnes();
            }
        }

        /// Copy camera state.
        viewer_camera = app_camera;
    }

    bool callbackKeyPressed(const PBSControlState &control_state, int key) override
    {
        switch (key)
        {
        case ImGuiKey_Space:
            m_simulating = !m_simulating;
            break;
        default:
            break;
        }
        return false;
    }

    bool callbackMouseMove(const PBSControlState &control_state, const Vector2F &mouse_pos,
                           const Vector2F &mouse_delta) override
    {
        if (control_state.modifiers[MOUSE_LEFT])
        {
            CameraHelper::panCameraFromDrag(app_camera, mouse_delta, CAMERA_PAN_SENSITIVITY);
        }

        return false;
    }

    bool callbackMouseScroll(const PBSControlState &control_state, float t) override
    {
        Vector3F camera_dist = app_camera.eye - app_camera.center;
        app_camera.eye = app_camera.center + camera_dist * (1.0 - CAMERA_ZOOM_SENSITIVITY * t);
        app_camera.height = app_camera.height * (1.0 - CAMERA_ZOOM_SENSITIVITY * t);

        return false;
    }

    bool step()
    {
        // apply source in density field
        p_density.applySource(0.45, 0.55, 0.1, 0.15);

        // external forces
        addBuoyancyForce();
        if (m_use_wind)
            addWindForce();
        applyForce();

        // remove divergence
        solvePressure();

        // advect everything
        advectValues();

        // reset forces
        p_force.reset();

        return true;
    }

    void solvePressure()
    {
        // apply boundary conditions
        setBoundaryConditions();

        // compute divergence
        computeDivergence();

        // solve Poisson equation
        solvePoisson();

        // velocity correction
        correctVelocity();

        // compute vorticity
        computeVorticity();

        // for debugging
        computeDivergence();
    }

    void setBoundaryConditions()
    {
        // x-velocity
        Array2d &u = p_velocity.x();
        int sx = u.size(0);
        int sy = u.size(1);
        for (int y = 0; y < sy; ++y)
        {
            // du/dx = 0 at left and right boundaries (Neumann BC)
            u(0, y) = u(2, y);
            u(sx - 1, y) = u(sx - 3, y);
        }
        for (int x = 0; x < sx; ++x)
        {
            // u = 0 at top and bottom boundaries (Dirichlet BC)
            u(x, 0) = 0;
            u(x, sy - 1) = 0;
        }

        // y-velocity
        Array2d &v = p_velocity.y();
        sx = v.size(0);
        sy = v.size(1);
        for (int x = 0; x < sx; ++x)
        {
            // dv/dy = 0 at top and bottom boundaries (Neumann BC)
            v(x, 0) = v(x, 2);
            v(x, sy - 1) = v(x, sy - 3);
        }
        sx = v.size(0);
        sy = v.size(1);
        for (int y = 0; y < sy; ++y)
        {
            // v = 0 at left and right boundaries (Dirichlet BC)
            v(0, y) = 0;
            v(sx - 1, y) = 0;
        }

        // pressure
        Array2d &p = p_pressure.x();
        sx = p.size(0);
        sy = p.size(1);
        for (int y = 0; y < sy; ++y)
        {
            // dp/dx = 0 at left and right boundaries (Neumann BC)
            p(0, y) = p(1, y);
            p(sx - 1, y) = p(sx - 2, y);
        }
        for (int x = 0; x < sx; ++x)
        {
            // dp/dy = 0 at top and bottom boundaries (Neumann BC)
            p(x, 0) = p(x, 1);
            p(x, sy - 1) = p(x, sy - 2);
        }
    }

    void computeDivergence()
    {
        // calculate divergence
        for (int y = 1; y < m_res_y - 1; ++y)
        {
            for (int x = 1; x < m_res_x - 1; ++x)
            {
                double dudx = (p_velocity.x()(x + 1, y) - p_velocity.x()(x, y)) / m_dx;
                double dvdy = (p_velocity.y()(x, y + 1) - p_velocity.y()(x, y)) / m_dx;
                // if (std::abs(dudx + dvdy) > 1e-10)
                //     std::cout << "divergence at (" << x << "," << y << "):" << dudx + dvdy << std::endl;

                p_divergence.x()(x, y) = dudx + dvdy;
            }
        }
    }

    void computeVorticity()
    {
        // calculate vorticity
        for (int y = 2; y < m_res_y - 2; ++y)
        {
            for (int x = 2; x < m_res_x - 2; ++x)
            {
                double dudy = (p_velocity.x()(x, y + 1) - p_velocity.x()(x, y - 1)) * 0.5 / m_dx;
                double dvdx = (p_velocity.y()(x + 1, y) - p_velocity.y()(x - 1, y)) * 0.5 / m_dx;
                p_vorticity.x()(x, y) = dvdx - dudy;
            }
        }
    }

public:
    // ExternalForce.cpp
    void addBuoyancyForce();
    void addWindForce();
    void applyForce();

public:
    // FluidSim.cpp
    void solvePoisson();
    void correctVelocity();
    void advectValues();
    void advectDensitySL(const Array2d &u, const Array2d &v);
    void advectVelocitySL(const Array2d &u, const Array2d &v);
    void MacCormackUpdate(const Array2d &d, const Array2d &d_forward, const Array2d &u, const Array2d &u_forward, const Array2d &v, const Array2d &v_forward);
    void MacCormackClamp(const Array2d &d, const Array2d &d_forward, const Array2d &u, const Array2d &u_forward, const Array2d &v, const Array2d &v_forward);

public:
    // Test.cpp
    std::string m_check();
};