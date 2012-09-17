@echo off

echo =======================================================
echo GENERATING SAMPLE BUFFER for reconstruction
echo =======================================================
pbrt-dsb-x64_Release.exe --pnggamma 1.6 --dumpsamples mon-8spp.pbrt

echo =======================================================
echo GENERATING RAY BUFFER for 128 SPP glossy reconstruction
echo =======================================================
pbrt-dsb-x64_Release.exe --pnggamma 1.6 --dumpreconstructionrays --reconstruct samplebuffer.bin mon-128spp.pbrt
move /Y rays.bin mon-rays-128spp.bin
rem the resulting images are garbage, delete them
del mon-gt-128spp.exr
del mon-gt-128spp-gamma16.png

echo =======================================================
echo GENERATING RAY BUFFER for 512 SPP glossy reconstruction
echo =======================================================
pbrt-dsb-x64_Release.exe --pnggamma 1.6 --dumpreconstructionrays --reconstruct samplebuffer.bin mon-512spp.pbrt
move /Y rays.bin mon-rays-512spp.bin
rem the resulting images are garbage, delete them
del mon-gt-512spp.exr
del mon-gt-512spp-gamma16.png

echo =======================================================
echo RENDERING GROUND TRUTH 512 SPP
echo =======================================================
pbrt-dsb-x64_Release.exe --pnggamma 1.6 --indirectonly mon-512spp.pbrt
