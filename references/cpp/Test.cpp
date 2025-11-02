#include "FluidApp.h"

#include <iostream>
#include <iomanip> 
#include <random>
#include <fstream>


class TestCase {
public:
    std::string test_name;
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

    Grid2 next_p_density;
    Grid2 next_p_pressure;
    Grid2 next_p_divergence;
    Grid2 next_p_vorticity;

    MACGrid2 next_p_velocity;
    MACGrid2 next_p_force;

    Grid2 forward_density;
    MACGrid2 forward_velocity;

    Grid2 backward_density;
    MACGrid2 backward_velocity;

    void save_grid_state(const Grid2 &grid, std::ostream &out) const {
        assert(grid.x().size(0) == m_res_x && grid.x().size(1) == m_res_y);
        for (int y = 0; y < grid.x().size(1); ++y)
            for (int x = 0; x < grid.x().size(0); ++x)
                out << grid.x()(x, y) << " ";
        out << " | ";
    }

    void load_grid_state(Grid2 &grid, std::istream &in, double &dx) {
        std::string buf;
        grid = Grid2(m_res_x, m_res_y, dx);
        for (int y = 0; y < m_res_y; ++y)
            for (int x = 0; x < m_res_x; ++x) {
                in >> grid.x()(x, y);
            }
        in >> buf;
    }

    void save_mac_grid_state(MACGrid2 &grid, std::ostream &out) const {
        assert(grid.x().size(0) == m_res_x + 1 && grid.x().size(1) == m_res_y);
        assert(grid.y().size(0) == m_res_x && grid.y().size(1) == m_res_y + 1);
        for (int y = 0; y < grid.x().size(1); ++y)
            for (int x = 0; x < grid.x().size(0); ++x)
                out << grid.x()(x, y) << " ";
        out << " | ";

        for (int y = 0; y < grid.y().size(1); ++y)
            for (int x = 0; x < grid.y().size(0); ++x)
                out << grid.y()(x, y) << " ";
        out << " | ";
    }

    void load_mac_grid_state(MACGrid2 &grid, std::istream &in, double &dx) {
        std::string buf;
        grid = MACGrid2(m_res_x, m_res_y, dx);
        for (int y = 0; y < m_res_y; ++y) {
            for (int x = 0; x < m_res_x + 1; ++x) {
                in >> grid.x()(x, y);
            }
        }
        in >> buf;
        for (int y = 0; y < m_res_y + 1; ++y) {
            for (int x = 0; x < m_res_x; ++x) {
                in >> grid.y()(x, y);
            }
        }
        in >> buf;
    }

    void save(std::ostream &out) {
        out << std::setprecision(17);
        out << test_name << " | ";
        out << m_dt << " | ";
        out << m_tolerance << " | ";
        out << m_max_iter << " | ";
        out << m_use_wind << " | ";
        out << m_use_maccormack << " | ";

        out << m_res_x << " | ";
        out << m_res_y << " | ";
        out << m_size_x << " | ";
        out << m_size_y << " | ";
        out << m_dx << " | ";

        save_grid_state(p_density, out);
        save_grid_state(p_pressure, out);
        save_grid_state(p_divergence, out);
        save_grid_state(p_vorticity, out);

        save_mac_grid_state(p_velocity, out);
        save_mac_grid_state(p_force, out);

        save_grid_state(next_p_density, out);
        save_grid_state(next_p_pressure, out);
        save_grid_state(next_p_divergence, out);
        save_grid_state(next_p_vorticity, out);

        save_mac_grid_state(next_p_velocity, out);
        save_mac_grid_state(next_p_force, out);

        if (test_name == "testMacCormackUpdate" || test_name == "testMacCormackClamp") {
            save_grid_state(forward_density, out);
            save_mac_grid_state(forward_velocity, out);
            save_grid_state(backward_density, out);
            save_mac_grid_state(backward_velocity, out);
        }

        out << std::endl;

    }

    bool load(const std::string& line) {
        std::istringstream in(line);
        std::string buf;
        in >> test_name;
        in >> buf;
        in >> m_dt;
        in >> buf;
        in >> m_tolerance;
        in >> buf;
        in >> m_max_iter;
        in >> buf;
        in >> m_use_wind;
        in >> buf;
        in >> m_use_maccormack;
        in >> buf;

        in >> m_res_x;
        in >> buf;
        in >> m_res_y;
        in >> buf;
        in >> m_size_x;
        in >> buf;
        in >> m_size_y;
        in >> buf;
        in >> m_dx;
        in >> buf;

        load_grid_state(p_density, in, m_dx);
        load_grid_state(p_pressure, in, m_dx);
        load_grid_state(p_divergence, in, m_dx);
        load_grid_state(p_vorticity, in, m_dx);

        load_mac_grid_state(p_velocity, in, m_dx);
        load_mac_grid_state(p_force, in, m_dx);

        load_grid_state(next_p_density, in, m_dx);
        load_grid_state(next_p_pressure, in, m_dx);
        load_grid_state(next_p_divergence, in, m_dx);
        load_grid_state(next_p_vorticity, in, m_dx);

        load_mac_grid_state(next_p_velocity, in, m_dx);
        load_mac_grid_state(next_p_force, in, m_dx);

        if (test_name == "testMacCormackUpdate" || test_name == "testMacCormackClamp") {
            load_grid_state(forward_density, in, m_dx);
            load_mac_grid_state(forward_velocity, in, m_dx);
            load_grid_state(backward_density, in, m_dx);
            load_mac_grid_state(backward_velocity, in, m_dx);
        }

        return in.good();

    }

    void initialize_app(FluidApp &app) const {
        app.m_dt = m_dt;
        app.m_tolerance = m_tolerance;
        app.m_max_iter = m_max_iter;
        app.m_use_wind = m_use_wind;
        app.m_use_maccormack = m_use_maccormack;

        app.m_res_x = m_res_x;
        app.m_res_y = m_res_y;
        app.m_size_x = m_size_x;
        app.m_size_y = m_size_y;
        app.m_dx = m_dx;

        app.p_density = p_density;
        app.p_pressure = p_pressure;
        app.p_divergence = p_divergence;
        app.p_vorticity = p_vorticity;

        app.p_velocity = p_velocity;
        app.p_force = p_force;

        // Get mesh for rendering from any grid.
        app.p_density.getMesh(app.render_V, app.render_F);
    }

    void init_from_app(const FluidApp &app) {
        m_dt = app.m_dt;
        m_tolerance = app.m_tolerance;
        m_max_iter = app.m_max_iter;
        m_use_wind = app.m_use_wind;
        m_use_maccormack = app.m_use_maccormack;

        m_res_x = app.m_res_x;
        m_res_y = app.m_res_y;
        m_size_x = app.m_size_x;
        m_size_y = app.m_size_y;
        m_dx = app.m_dx;
    }

    void save_initial_state(const FluidApp &app) {
        p_density = app.p_density;
        p_pressure = app.p_pressure;
        p_divergence = app.p_divergence;
        p_vorticity = app.p_vorticity;

        p_velocity = app.p_velocity;
        p_force = app.p_force;
    }

    void save_next_step_state(const FluidApp &app) {
        next_p_density = app.p_density;
        next_p_pressure = app.p_pressure;
        next_p_divergence = app.p_divergence;
        next_p_vorticity = app.p_vorticity;

        next_p_velocity = app.p_velocity;
        next_p_force = app.p_force;
    }

    void save_forward_state(const FluidApp &app) {
        forward_density = app.p_density;
        forward_velocity = app.p_velocity;
    }

    void save_backward_state(const FluidApp &app) {
        backward_density = app.p_density;
        backward_velocity = app.p_velocity;
    }

    FluidApp init_test(bool m_use_maccormack) {
        FluidApp mockApp;
        mockApp.m_use_maccormack = m_use_maccormack;
        mockApp.m_res_x = 20;
        mockApp.m_res_y = 30;
        mockApp.resetSimulation();
        for (int i = 0; i < 50; ++i) {
            mockApp.step();
        }
        init_from_app(mockApp);
        return mockApp;
    }
};


std::string testSolvePoisson(std::vector<TestCase>& test_case) {
    for (auto& test_case : test_case) {
        FluidApp app;
        test_case.initialize_app(app);
        app.solvePoisson();
        Grid2& pressure = app.p_pressure;
        for (int y = 0; y < pressure.x().size(1); ++y) {
            for (int x = 0; x < pressure.x().size(0); ++x) {
                if (std::abs(pressure.x()(x, y) - test_case.next_p_pressure.x()(x, y)) > 1e-5) {
                    std::string output = "solvePoisson FAILED:\n";
                    output += "Pressure at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(test_case.next_p_pressure.x()(x, y)) + ", Got: " + std::to_string(pressure.x()(x, y)) + "\n";
                    return output;
                }
            }
        }
    }
    return "solvePoisson: PASSED\n";
}


std::string testCorrectVelocity(std::vector<TestCase>& test_case) {
    for (auto& test_case : test_case) {
        FluidApp app;
        test_case.initialize_app(app);
        app.correctVelocity();
        MACGrid2& predicted_velocity = app.p_velocity;
        MACGrid2& expected_velocity = test_case.next_p_velocity;
        for (int y = 0; y < predicted_velocity.x().size(1); ++y) {
            for (int x = 0; x < predicted_velocity.x().size(0); ++x) {
                if (std::abs(predicted_velocity.x()(x, y) - expected_velocity.x()(x, y)) > 1e-5) {
                    std::string output = "correctVelocity FAILED:\n";
                    output += "Velocity x at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_velocity.x()(x, y)) + ", Got: " + std::to_string(predicted_velocity.x()(x, y)) + "\n";
                    return output;
                }
            }
        }
        for (int y = 0; y < predicted_velocity.y().size(1); ++y) {
            for (int x = 0; x < predicted_velocity.y().size(0); ++x) {
                if (std::abs(predicted_velocity.y()(x, y) - expected_velocity.y()(x, y)) > 1e-5) {
                    std::string output = "correctVelocity FAILED:\n";
                    output += "Velocity y at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_velocity.y()(x, y)) + ", Got: " + std::to_string(predicted_velocity.y()(x, y)) + "\n";
                    return output;
                }
            }
        }
    }
    return "correctVelocity: PASSED\n";
}


std::string testAdvectValues(std::vector<TestCase>& test_case) {
    for (auto& test_case : test_case) {
        FluidApp app;
        test_case.initialize_app(app);
        app.advectValues();
        
        Grid2& predicted_density = app.p_density;
        Grid2& expected_density = test_case.next_p_density;

        MACGrid2& predicted_velocity = app.p_velocity;
        MACGrid2& expected_velocity = test_case.next_p_velocity;

        for (int y = 0; y < predicted_density.x().size(1); ++y) {
            for (int x = 0; x < predicted_density.x().size(0); ++x) {
                if (std::abs(predicted_density.x()(x, y) - expected_density.x()(x, y)) > 1e-5) {
                    std::string output = "advectValues FAILED:\n";
                    output += "Density at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_density.x()(x, y)) + ", Got: " + std::to_string(predicted_density.x()(x, y)) + "\n";
                    return output;
                }
            }
        }
        for (int y = 0; y < predicted_velocity.x().size(1); ++y) {
            for (int x = 0; x < predicted_velocity.x().size(0); ++x) {
                if (std::abs(predicted_velocity.x()(x, y) - expected_velocity.x()(x, y)) > 1e-5) {
                    std::string output = "advectValues FAILED:\n";
                    output += "Velocity x at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_velocity.x()(x, y)) + ", Got: " + std::to_string(predicted_velocity.x()(x, y)) + "\n";
                    return output;
                }
            }
        }
        for (int y = 0; y < predicted_velocity.y().size(1); ++y) {
            for (int x = 0; x < predicted_velocity.y().size(0); ++x) {
                if (std::abs(predicted_velocity.y()(x, y) - expected_velocity.y()(x, y)) > 1e-5) {
                    std::string output = "advectValues FAILED:\n";
                    output += "Velocity y at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_velocity.y()(x, y)) + ", Got: " + std::to_string(predicted_velocity.y()(x, y)) + "\n";
                    return output;
                }
            }
        }
    }
    return "advectValues: PASSED\n";
}


std::string testAdvectDensitySL(std::vector<TestCase>& test_case) {
    for (auto& test_case : test_case) {
        FluidApp app;
        test_case.initialize_app(app);
        app.advectDensitySL(app.p_velocity.x(), app.p_velocity.y());
        Grid2& predicted_density = app.p_density;
        Grid2& expected_density = test_case.next_p_density;

        for (int y = 0; y < predicted_density.x().size(1); ++y) {
            for (int x = 0; x < predicted_density.x().size(0); ++x) {
                if (std::abs(predicted_density.x()(x, y) - expected_density.x()(x, y)) > 1e-5) {
                    std::string output = "advectValues FAILED:\n";
                    output += "Density at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_density.x()(x, y)) + ", Got: " + std::to_string(predicted_density.x()(x, y)) + "\n";
                    return output;
                }
            }
        }
    }
    return "advectDensitySL: PASSED\n";
}


std::string testAdvectVelocitySL(std::vector<TestCase>& test_case) {
    for (auto& test_case : test_case) {
        FluidApp app;
        test_case.initialize_app(app);
        app.advectVelocitySL(app.p_velocity.x(), app.p_velocity.y());
        MACGrid2& predicted_velocity = app.p_velocity;
        MACGrid2& expected_velocity = test_case.next_p_velocity;
        for (int y = 0; y < predicted_velocity.x().size(1); ++y) {
            for (int x = 0; x < predicted_velocity.x().size(0); ++x) {
                if (std::abs(predicted_velocity.x()(x, y) - expected_velocity.x()(x, y)) > 1e-5) {
                    std::string output = "advectValues FAILED:\n";
                    output += "Velocity x at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_velocity.x()(x, y)) + ", Got: " + std::to_string(predicted_velocity.x()(x, y)) + "\n";
                    return output;
                }
            }
        }
        for (int y = 0; y < predicted_velocity.y().size(1); ++y) {
            for (int x = 0; x < predicted_velocity.y().size(0); ++x) {
                if (std::abs(predicted_velocity.y()(x, y) - expected_velocity.y()(x, y)) > 1e-5) {
                    std::string output = "advectValues FAILED:\n";
                    output += "Velocity y at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_velocity.y()(x, y)) + ", Got: " + std::to_string(predicted_velocity.y()(x, y)) + "\n";
                    return output;
                }
            }
        }
    }
    return "advectVelocitySL: PASSED\n";
}


std::string testMacCormackUpdate(std::vector<TestCase>& test_case) {
   for (auto& test_case : test_case) {
        FluidApp app;
        test_case.initialize_app(app);
        Array2d d = test_case.backward_density.x();
        Array2d u = test_case.backward_velocity.x();
        Array2d v = test_case.backward_velocity.y();
        Array2d forward_d = test_case.forward_density.x();
        Array2d forward_u = test_case.forward_velocity.x();
        Array2d forward_v = test_case.forward_velocity.y();
        
        app.MacCormackUpdate(d, forward_d, u, forward_u, v, forward_v);
        
        Grid2& predicted_density = app.p_density;
        Grid2& expected_density = test_case.next_p_density;

        MACGrid2& predicted_velocity = app.p_velocity;
        MACGrid2& expected_velocity = test_case.next_p_velocity;

        for (int y = 0; y < predicted_density.x().size(1); ++y) {
            for (int x = 0; x < predicted_density.x().size(0); ++x) {
                if (std::abs(predicted_density.x()(x, y) - expected_density.x()(x, y)) > 1e-5) {
                    std::string output = "macCormackUpdate FAILED:\n";
                    output += "Density at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_density.x()(x, y)) + ", Got: " + std::to_string(predicted_density.x()(x, y)) + "\n";
                    return output;
                }
            }
        }
        for (int y = 0; y < predicted_velocity.x().size(1); ++y) {
            for (int x = 0; x < predicted_velocity.x().size(0); ++x) {
                if (std::abs(predicted_velocity.x()(x, y) - expected_velocity.x()(x, y)) > 1e-5) {
                    std::string output = "macCormackUpdate FAILED:\n";
                    output += "Velocity x at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_velocity.x()(x, y)) + ", Got: " + std::to_string(predicted_velocity.x()(x, y)) + "\n";
                    return output;
                }
            }
        }
        for (int y = 0; y < predicted_velocity.y().size(1); ++y) {
            for (int x = 0; x < predicted_velocity.y().size(0); ++x) {
                if (std::abs(predicted_velocity.y()(x, y) - expected_velocity.y()(x, y)) > 1e-5) {
                    std::string output = "macCormackUpdate FAILED:\n";
                    output += "Velocity y at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_velocity.y()(x, y)) + ", Got: " + std::to_string(predicted_velocity.y()(x, y)) + "\n";
                    return output;
                }
            }
        }
    }
    return "macCormackUpdate: PASSED\n";
}


std::string testMacCormackClamp(std::vector<TestCase>& test_case) {
    for (auto& test_case : test_case) {
        FluidApp app;
        test_case.initialize_app(app);
        Array2d d = test_case.backward_density.x();
        Array2d u = test_case.backward_velocity.x();
        Array2d v = test_case.backward_velocity.y();
        Array2d forward_d = test_case.forward_density.x();
        Array2d forward_u = test_case.forward_velocity.x();
        Array2d forward_v = test_case.forward_velocity.y();

        app.MacCormackClamp(d, forward_d, u, forward_u, v, forward_v);

        Grid2& predicted_density = app.p_density;
        Grid2& expected_density = test_case.next_p_density;

        MACGrid2& predicted_velocity = app.p_velocity;
        MACGrid2& expected_velocity = test_case.next_p_velocity;

        for (int y = 0; y < predicted_density.x().size(1); ++y) {
            for (int x = 0; x < predicted_density.x().size(0); ++x) {
                if (std::abs(predicted_density.x()(x, y) - expected_density.x()(x, y)) > 1e-5) {
                    std::string output = "macCormackClamp FAILED:\n";
                    output += "Density at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_density.x()(x, y)) + ", Got: " + std::to_string(predicted_density.x()(x, y)) + "\n";
                    return output;
                }
            }
        }
        for (int y = 0; y < predicted_velocity.x().size(1); ++y) {
            for (int x = 0; x < predicted_velocity.x().size(0); ++x) {
                if (std::abs(predicted_velocity.x()(x, y) - expected_velocity.x()(x, y)) > 1e-5) {
                    std::string output = "macCormackClamp FAILED:\n";
                    output += "Velocity x at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_velocity.x()(x, y)) + ", Got: " + std::to_string(predicted_velocity.x()(x, y)) + "\n";
                    return output;
                }
            }
        }
        for (int y = 0; y < predicted_velocity.y().size(1); ++y) {
            for (int x = 0; x < predicted_velocity.y().size(0); ++x) {
                if (std::abs(predicted_velocity.y()(x, y) - expected_velocity.y()(x, y)) > 1e-5) {
                    std::string output = "macCormackClamp FAILED:\n";
                    output += "Velocity y at (" + std::to_string(x) + ", " + std::to_string(y) + "):\n";
                    output += "Expected: " + std::to_string(expected_velocity.y()(x, y)) + ", Got: " + std::to_string(predicted_velocity.y()(x, y)) + "\n";
                    return output;
                }
            }
        }
    }
    return "macCormackClamp: PASSED\n";
}


std::map<std::string, std::function<std::string(std::vector<TestCase>&)>> tests = {
    {"testSolvePoisson", testSolvePoisson},
    {"testCorrectVelocity", testCorrectVelocity},
    {"testAdvectValues", testAdvectValues},
    {"testAdvectDensitySL", testAdvectDensitySL},
    {"testAdvectVelocitySL", testAdvectVelocitySL},
    {"testMacCormackUpdate", testMacCormackUpdate},
    {"testMacCormackClamp", testMacCormackClamp}
};


void generateFluidTests() {

    std::vector<TestCase> test_cases;

    FluidApp mockApp;
    Array2d u;
    Array2d v;
    Array2d d;
    Array2d d_forward;
    Array2d u_forward;
    Array2d v_forward;

    TestCase solve_poisson_test_case;
    solve_poisson_test_case.test_name = "testSolvePoisson";
    mockApp = solve_poisson_test_case.init_test(false);
    solve_poisson_test_case.save_initial_state(mockApp);
    mockApp.solvePoisson();
    solve_poisson_test_case.save_next_step_state(mockApp);
    test_cases.push_back(solve_poisson_test_case);

    TestCase correct_velocity_test_case;
    correct_velocity_test_case.test_name = "testCorrectVelocity";
    mockApp = correct_velocity_test_case.init_test(false);
    mockApp.solvePoisson();
    correct_velocity_test_case.save_initial_state(mockApp);
    mockApp.correctVelocity();
    correct_velocity_test_case.save_next_step_state(mockApp);
    test_cases.push_back(correct_velocity_test_case);
    

    TestCase advect_values_test_case;
    advect_values_test_case.test_name = "testAdvectValues";
    mockApp = advect_values_test_case.init_test(true);
    mockApp.p_density.applySource(0.45, 0.55, 0.1, 0.15);
    mockApp.addBuoyancyForce();
    if (mockApp.m_use_wind)
        mockApp.addWindForce();
    mockApp.applyForce();
    mockApp.solvePressure();
    advect_values_test_case.save_initial_state(mockApp);
    mockApp.advectValues();
    advect_values_test_case.save_next_step_state(mockApp);
    test_cases.push_back(advect_values_test_case);

    TestCase advect_density_sl_test_case;
    advect_density_sl_test_case.test_name = "testAdvectDensitySL";
    mockApp = advect_density_sl_test_case.init_test(false);
    mockApp.p_density.applySource(0.45, 0.55, 0.1, 0.15);
    mockApp.addBuoyancyForce();
    if (mockApp.m_use_wind)
        mockApp.addWindForce();
    mockApp.applyForce();
    mockApp.solvePressure();
    advect_density_sl_test_case.save_initial_state(mockApp);
    u = Array2d(mockApp.p_velocity.x());
    v = Array2d(mockApp.p_velocity.y());
    mockApp.advectDensitySL(u, v);
    advect_density_sl_test_case.save_next_step_state(mockApp);
    test_cases.push_back(advect_density_sl_test_case);

    TestCase advect_velocity_sl_test_case;
    advect_velocity_sl_test_case.test_name = "testAdvectVelocitySL";
    mockApp = advect_velocity_sl_test_case.init_test(false);
    mockApp.p_density.applySource(0.45, 0.55, 0.1, 0.15);
    mockApp.addBuoyancyForce();
    if (mockApp.m_use_wind)
        mockApp.addWindForce();
    mockApp.applyForce();
    mockApp.solvePressure();
    advect_velocity_sl_test_case.save_initial_state(mockApp);
    u = Array2d(mockApp.p_velocity.x());
    v = Array2d(mockApp.p_velocity.y());
    mockApp.advectVelocitySL(u, v);
    advect_velocity_sl_test_case.save_next_step_state(mockApp);
    test_cases.push_back(advect_velocity_sl_test_case);

    TestCase mac_cormack_update_test_case;
    mac_cormack_update_test_case.test_name = "testMacCormackUpdate";
    mockApp = mac_cormack_update_test_case.init_test(true);
    mockApp.p_density.applySource(0.45, 0.55, 0.1, 0.15);
    mockApp.addBuoyancyForce();
    if (mockApp.m_use_wind)
        mockApp.addWindForce();
    mockApp.applyForce();
    mockApp.solvePressure();
    d = Array2d(mockApp.p_density.x());
    u = Array2d(mockApp.p_velocity.x());
    v = Array2d(mockApp.p_velocity.y());
    mac_cormack_update_test_case.save_backward_state(mockApp);
    mockApp.advectDensitySL(u, v);
    mockApp.advectVelocitySL(u, v);
    d_forward = Array2d(mockApp.p_density.x());
    u_forward = Array2d(mockApp.p_velocity.x());
    v_forward = Array2d(mockApp.p_velocity.y());
    mac_cormack_update_test_case.save_forward_state(mockApp);
    mac_cormack_update_test_case.save_initial_state(mockApp);
    mockApp.MacCormackUpdate(d, d_forward, u, u_forward, v, v_forward);
    mac_cormack_update_test_case.save_next_step_state(mockApp);
    test_cases.push_back(mac_cormack_update_test_case);

    TestCase mac_cormack_clamp_test_case;
    mac_cormack_clamp_test_case.test_name = "testMacCormackClamp";
    mockApp = mac_cormack_clamp_test_case.init_test(true);
    mockApp.p_density.applySource(0.45, 0.55, 0.1, 0.15);
    mockApp.addBuoyancyForce();
    if (mockApp.m_use_wind)
        mockApp.addWindForce();
    mockApp.applyForce();
    mockApp.solvePressure();
    d = Array2d(mockApp.p_density.x());
    u = Array2d(mockApp.p_velocity.x());
    v = Array2d(mockApp.p_velocity.y());
    mac_cormack_clamp_test_case.save_backward_state(mockApp);
    mockApp.advectDensitySL(u, v);
    mockApp.advectVelocitySL(u, v);
    d_forward = Array2d(mockApp.p_density.x());
    u_forward = Array2d(mockApp.p_velocity.x());
    v_forward = Array2d(mockApp.p_velocity.y());
    mac_cormack_clamp_test_case.save_forward_state(mockApp);
    mockApp.MacCormackUpdate(d, d_forward, u, u_forward, v, v_forward);
    mac_cormack_clamp_test_case.save_initial_state(mockApp);
    mockApp.MacCormackClamp(d, d_forward, u, u_forward, v, v_forward);
    mac_cormack_clamp_test_case.save_next_step_state(mockApp);
    test_cases.push_back(mac_cormack_clamp_test_case);

    std::ofstream fout("fluidTests.dat");
    if (!fout) {
        throw std::runtime_error("Could not open file!");
    }
    for (auto& test_case : test_cases) {
        test_case.save(fout);
    }

    fout.close();

    std::cout << "Test cases generated and saved to 'fluidTests.dat'" << std::endl;
    
}

std::string FluidApp::m_check() {
    std::string check_info;

    // generateFluidTests();
    
    std::map<std::string, std::vector<TestCase>> test_cases;
    std::ifstream fin("3_fluid/TestCases.dat");
    if (!fin) {
        return "ERROR: Could not open test file '3_fluid/TestCases.dat'!";
    }
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(fin, line)) {
        lines.push_back(line);
        TestCase test_case;
        if (test_case.load(line)) {
            std::string test_name = test_case.test_name;
            test_cases[test_name].push_back(test_case);
        } else {
            check_info += "Failed to load test line: " + line + "\n";
        }
    }
    fin.close();

    std::cout << "Loaded " << lines.size() << " test cases from 'fluidTests.dat'" << std::endl;

    for (auto &test : tests) {
        const std::string &test_name = test.first;
        auto it = test_cases.find(test_name);
        if (it != test_cases.end()) {
            check_info += test.second(it->second);
        }
    }

    return check_info;
}