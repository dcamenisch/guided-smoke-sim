#include "FluidApp.h"

void FluidApp::addBuoyancyForce()
{
    double scaling = 64.0 / m_res_x;

    // add buoyancy
    for (int i = 0; i < p_force.y().size(0); ++i)
    {
        for (int j = 1; j < p_force.y().size(1) - 1; ++j)
        {
            p_force.y()(i, j) += 0.1 * (p_density.x()(i, j - 1) + p_density.x()(i, j)) / 2.0 * scaling;
        }
    }
}

void FluidApp::addWindForce()
{
    double scaling = 64.0 / m_res_x;

    static double r = 0.0;
    r += 1;

    const double fx = 2e-2 * cos(5e-2 * r) * cos(3e-2 * r) * scaling;

    // add wind
    for (int i = 0; i < p_force.x().size(0); ++i)
    {
        for (int j = 0; j < p_force.x().size(1); ++j)
        {
            p_force.x()(i, j) += fx;
        }
    }
}

void FluidApp::applyForce()
{
    for (int i = 0; i < p_velocity.x().size(0); ++i)
    {
        for (int j = 0; j < p_velocity.x().size(1); ++j)
        {
            p_velocity.x()(i, j) += m_dt * p_force.x()(i, j);
        }
    }

    for (int i = 0; i < p_velocity.y().size(0); ++i)
    {
        for (int j = 0; j < p_velocity.y().size(1); ++j)
        {
            p_velocity.y()(i, j) += m_dt * p_force.y()(i, j);
        }
    }
}