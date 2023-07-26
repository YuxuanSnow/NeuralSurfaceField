#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                                                     #
#                                                             conditional sdf to learn coarse template                                                                #
#                                                                                                                                                                     #
#                                                                                                                                                                     #
class Basic_Trainer_invskinning(object):

    def __init__(self,  module_dict, device, dataset):
        self.device = device

        if module_dict.get('diffused_skinning_field') is not None:
            self.diffused_skinning_field = module_dict.get('diffused_skinning_field').to(self.device)

        if module_dict.get('inv_skinner') is not None:
            self.inv_skinner = module_dict.get('inv_skinner').to(self.device)
            self.inv_skinner_normal = module_dict.get('inv_skinner_normal').to(self.device)

        if module_dict.get('skinner') is not None:
            self.skinner = module_dict.get('skinner').to(self.device)
            self.skinner_normal = module_dict.get('skinner_normal').to(self.device)

        self.dataset = dataset
#                                                                                                                                                                     #
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#