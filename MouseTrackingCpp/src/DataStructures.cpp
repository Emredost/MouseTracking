#include "DataStructures.h"

// Implementation file for DataStructures
// Most functionality is implemented in the header as templates and inline functions
// This file is here to ensure proper compilation and any future implementations

// Explicit template instantiations for better compilation performance
template class ThreadSafeVector<MouseEvent>;
template class ThreadSafeVector<GazeEvent>;
template class ThreadSafeVector<SyncEvent>; 