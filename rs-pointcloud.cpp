// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "example.hpp"          // Include short list of convenience functions for rendering
#include <librealsense2/hpp/rs_export.hpp>
#include <algorithm>            // std::min, std::max
#include <librealsense2/rsutil.h>
#include <math.h>

// Helper functions
void register_glfw_callbacks(window& app, glfw_state& app_state);


int main(int argc, char * argv[]) try
{
    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense Pointcloud Example");
    // Construct an object to manage view state
    glfw_state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;
    rs2::colorizer colorizer;
    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();

    while (app) // Application still alive?
    {
        // Wait for the next set of frames from the camera
        auto frames = pipe.wait_for_frames();

        rs2::frame color;//auto color = frames.get_color_frame();
        color = frames.get_color_frame();

        // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
        if (!color)
            color = frames.get_infrared_frame();

        // Tell pointcloud object to map to this color frame
        pc.map_to(color);

        auto depth = frames.get_depth_frame();

        // Generate the pointcloud and texture mappings
        points = pc.calculate(depth);

        // Upload the color frame to OpenGL
        app_state.tex.upload(color);
        rs2::frame colorized;
        colorized = colorizer.process(frames);
        // Draw the pointcloud
        draw_pointcloud(app.width(), app.height(), app_state, points);
       /*
        points.export_to_ply("anotherface2.ply", color);

        rs2::save_to_ply exporter("meshmyface2.ply", pc);
        exporter.set_option(rs2::save_to_ply::OPTION_PLY_MESH, 1.f);
        exporter.set_option(rs2::save_to_ply::OPTION_PLY_NORMALS, 1.f);
        exporter.set_option(rs2::save_to_ply::OPTION_PLY_BINARY, 0.f);
        exporter.process(color);*/
        
    }
    if (!app) {
        // Wait for the next set of frames from the camera
        auto frames = pipe.wait_for_frames();

        rs2::frame color;
        color = frames.get_color_frame();//auto color = frames.get_color_frame();

        // Tell pointcloud object to map to this color frame
        pc.map_to(color);

        auto depth = frames.get_depth_frame();

        // Generate the pointcloud and texture mappings
        points = pc.calculate(depth);

        points.export_to_ply("anotherface5.ply", color);

        rs2::frame colorized;
        colorized = colorizer.process(frames);
        
        rs2::save_to_ply exporter("mesh_faac4.ply", pc);
        exporter.set_option(rs2::save_to_ply::OPTION_PLY_MESH, 1.f);
        exporter.set_option(rs2::save_to_ply::OPTION_PLY_NORMALS, 1.f);
        exporter.set_option(rs2::save_to_ply::OPTION_PLY_BINARY, 0.f);
        //exporter.invoke(depth);
        exporter.process(colorized);
        //see how the point is transfered from pixel to point in 3Dcamera 

        float upixel[2]; // From pixel
        float upoint[3]; // From point (in 3D)

        float vpixel[2]; // To pixel
        float vpoint[3]; // To point (in 3D)
        auto data = points.get_vertices();
        auto x = data->x;
        auto y = data->y;
        std::cerr << "================THE POINT IS ================\n";
        std::cerr << x << std::endl;
        std::cerr << y << std::endl;
        std::cerr << "================================\n";
        /*
        upixel[0] = static_cast<float>();
        upixel[1] = static_cast<float>(u.second);
        vpixel[0] = static_cast<float>(v.first);
        vpixel[1] = static_cast<float>(v.second);

        auto udist = depth.get_distance(static_cast<int>(upixel[0]), static_cast<int>(upixel[1]));
        auto vdist = depth.get_distance(static_cast<int>(vpixel[0]), static_cast<int>(vpixel[1]));

        rs2_intrinsics intr = depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics(); // Calibration data
        rs2_deproject_pixel_to_point(upoint, &intr, upixel, udist);
        rs2_deproject_pixel_to_point(vpoint, &intr, vpixel, vdist);
        */
    }

    //pipe.stop();
    //ADD on for coordinate change from pixel coordinate to camera cooridnate 
    //rs2::config cfg;
    //cfg.enable_stream(RS2_STREAM_DEPTH);//Enable default depth 
    //For thecolor stream,set fromat to RGBA to allow blending of the color fram on top of the depth frame 
    //cfg.enable_stream(RS2_STREAM_COLOR, RS2_FORMAT_BGRA8);
    //auto profile = pipe.start(cfg);
    //rs2::spatial_filter spat;
    //rs2::temporal_filter temp;
    //rs2::align align_to(RS2_STREAM_DEPTH);
    //rs2::frameset current_frameset;

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
