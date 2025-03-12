#!/usr/bin/env python3

#############################################

# Base class for Transition
# This class is a base class for Transitions.

#############################################

from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
