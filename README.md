This code use OpenCV to replicate the result of Matlab code released by original authors of [paper](http://people.csail.mit.edu/nwadhwa/phase-video/).

Currently, it has components for motion magnification, with some details not fullfilled, like frequency attenuations.

The biggest problem is that although the aim of re-writing is to achieve real-time processing as frame incoming, the computation cost is considerably greater than Eulerian magnification.

TODOs:\
- [ ] Enable multi-threading.\
- [ ] Use GPU-API.
