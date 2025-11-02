#include "PBSApp.h"
#include "FluidApp.h"

int main(int argc, char *argv[])
{
    FluidApp app;
    PBSApp pbsapp(std::make_shared<FluidApp>(app));
    pbsapp.launch();

    return 0;
}