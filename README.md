# bytetrack-standalone

[ByteTrack](https://github.com/ifzhang/ByteTrack) is a simple and amazing new popular SOTA method. The source code
provided by the author are very insightful from research to deployment, like tensorrt, onnx, deepstream, ncpp.

One challenge I met with the official repo is that it is quite coupled with YoloX. 
In the meanwhile, what I need is a just simple standalone ByteTracker -- ready to use, standalone and as less dependencies as possible.

Thanks for the authors of [ByteTrack](https://github.com/ifzhang/ByteTrack), who already provides quite modularized code, 
I just simply further extract and made minor code (not logics) refactoring for this standalone bytetrack.

All dependencies are listed in the [requirements.txt](requirements.txt). 

### Run the Example

An example with mock videos and detectors is shown to illustrate how to use it.

From scratch, this is basically what you need

1. `python -m pip install Cython` (this step can be ignored if you already have Cython installed)
2. `pip install -r requirements.txt`
3. `python example.py`


### Marjor Differences Made

1. Make it torch independent
2. Remove all the args and make the hyper-parameters explicit
3. Clean up some not used imports
4. Clean up the imports
5. Renamed basetrack.py to base_track.py to keep names consistent

As you can see, it is more coding style and nothing to do with the logics.

