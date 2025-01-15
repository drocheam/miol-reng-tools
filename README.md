# miol-reng-tools
## Tools for Reverse Engineering Multifocal Intraocular Lenses

Repository of the project *Reverse Engineering of Multifocal Intraocular Lenses Using a Confocal Microscope* at TH Köln University of Applied Sciences.

An overview of the processing, measurements and results are featured in:<br>
> *Mendroch, D., Altmeyer, S. & Oberheide, U.; Characterization of diffractive bifocal intraocular lenses. Sci Rep 13, 908 (2023).* https://doi.org/10.1038/s41598-023-27521-7

The same processing has been applied to EDoF (Extended-Depth-of-Focus) intraocular lenses in the following publication:
> *Mendroch, D., Oberheide, U.,  Altmeyer, S.; Functional Design Analysis of Two Current Extended-Depth-of-Focus Intraocular Lenses. Trans. Vis. Sci. Tech. 2024;13(8):33.* https://doi.org/10.1167/tvst.13.8.33.

### Screenshots
 <img src="https://github.com/drocheam/miol-reng-tools/blob/main/Screenshot2.png" height="220">  <img src="https://github.com/drocheam/miol-reng-tools/blob/main/Screenshot1.png" height="220">  

### Structure

Element | Function
------------ | -------------
`Measurements\` | Sample measurement data
`Code\lib\` | Implemented library package
`Code\GenerateProfiles.py` | Generate surface profiles from a measurement
`Code\ShowProfiles.py` |  Show processed profiles

The project includes functions from the [musurf-reader](https://github.com/drocheam/musurf-reader) repository.
<br></br>

### Required packages:
`matplotlib, numpy, scipy`

Tested with Python 3.10

### Usage
Measurement files from a nanofocus µsurf custom confocal microscope are required (smt and smi format). 
A sample measurement can be found under `Measurements\`.
