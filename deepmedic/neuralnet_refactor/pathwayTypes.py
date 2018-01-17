#################################################################
#                        Types of Pathways                      #
#################################################################

# Also see module deepmedic.neuralnet.pathways.


class PathwayTypes(object):
    NORM = 0
    SUBS = 1
    FC = 2  # static

    def p_types(self):  # To iterate over if needed.
        # This enumeration is also the index in various data structures ala: [[listForNorm], [listForSubs], [listForFc]]
        return [self.NORM, self.SUBS, self.FC]
