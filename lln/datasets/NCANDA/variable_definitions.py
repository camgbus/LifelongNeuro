"""Definitions of relevant variables for the NCANDA data.
"""

import os
from itertools import product, combinations
from lln.local.paths import PATHS

VARS = {"tabular": {}, "structural": {}, "functional": {}, "diffusion": {}}
PATHS = {"tabular": os.path.join(PATHS["NCANDA_data"], "NCANDA_SNAPS_8Y_REDCAP_V03", "summaries", "redcap"),
         "structural": os.path.join(PATHS["NCANDA_data"], "NCANDA_SNAPS_8Y_STRUCTURAL_V01", "summaries", "structural", "longitudinal", "freesurfer"), 
         "functional": os.path.join(PATHS["NCANDA_data"], "NCANDA_SNAPS_8Y_RESTINGSTATE_V01", "summaries", "restingstate"), 
         "diffusion": os.path.join(PATHS["NCANDA_data"], "NCANDA_SNAPS_8Y_DIFFUSION_V01", "summaries", "restingstate")}

VARS["tabular"]['demographics'] = [("demographics", [('sex', 'sex_at_birth'), 
                                  ('family_id', 'family_id'), 
                                  ('race_label', 'race_ethnicity'), 
                                  ('hispanic', 'hispanic')])]

VARS["tabular"]['acquisition'] = [("demographics", [('visit_date', 'visit_date'),
                                 ('visit_age', 'age_years'), 
                                 ('site', 'site_id'), 
                                 ('scanner', 'scanner_manufacturer'), 
                                 ('scanner_model', 'scanner_model')])]

VARS["tabular"]['alcohol_initiation'] = [("youthreport1", [
       ("youthreport1_cddr15", "alcohol_age_first"), 
       ("youthreport1_cddr16", "alcohol_age_regular"),
       ("youthreport1_cddr19", "alcohol_nr_lifetime"),
       ("youthreport1_cddr21", "alcohol_nr_past_year"),
       ("youthreport1_cddr22", "alcohol_nr_past_month"),
       ("youthreport1_cddr27", "binge_drinking_ever"),
       ("youthreport1_cddr32", "binge_drinking_regular_ever"),
       ("youthreport1_cddr29", "binge_drinking_age_first"),
       ("youthreport1_cddr33", "binge_drinking_age_regular"),
       ("youthreport1_cddr30", "binge_drinking_nr_past_year"),
       ("youthreport1_cddr31", "binge_drinking_nr_past_month")
       ])]
             
VARS["tabular"]['marijuana_initiation'] = [("youthreport1", [
       ("youthreport1_cddr52", "marijuana_age_first"),
       ("youthreport1_cddr53", "marijuana_age_regular"),
       ("youthreport1_cddr55a", "marijuana_nr_lifetime"),
       ("youthreport1_cddr56", "marijuana_nr_past_year"),
       ("youthreport1_cddr57", "marijuana_nr_past_month"),
       ])]
                   
VARS["tabular"]['ecstasy_initiation'] = [("youthreport1", [
       ("youthreport1_cddr107", "ecstasy_age_first"),
       ("youthreport1_cddr108", "ecstasy_age_regular"),
       ("youthreport1_cddr109a", "ecstasy_nr_lifetime"),
       ("youthreport1_cddr112a", "ecstasy_nr_past_year"),
       ("youthreport1_cddr112b", "ecstasy_nr_past_month"),
       ])]

VARS["tabular"]['hallu_initiation'] = [("youthreport1", [
       ("youthreport1_cddr83", "hallu_age_first"),
       ("youthreport1_cddr84", "hallu_age_regular"),
       ("youthreport1_cddr85a", "hallu_nr_lifetime"),
       ("youthreport1_cddr87", "hallu_nr_past_year"),
       ("youthreport1_cddr88", "hallu_nr_past_month"),
       ])]


VARS["tabular"]['ketamine_initiation'] = [("youthreport1", [
       ("youthreport1_cddr113", "ketamine_age_first"),
       ("youthreport1_cddr114", "ketamine_age_regular"),
       ("youthreport1_cddr115a", "ketamine_nr_lifetime"),
       ("youthreport1_cddr117", "ketamine_nr_past_year"),
       ("youthreport1_cddr118", "ketamine_nr_past_month"),
       ])]

VARS["tabular"]['cocaine_initiation'] = [("youthreport1", [
       ("youthreport1_cddr89", "cocaine_age_first"),
       ("youthreport1_cddr90", "cocaine_age_regular"),
       ("youthreport1_cddr91a", "cocaine_nr_lifetime"),
       ("youthreport1_cddr93", "cocaine_nr_past_year"),
       ("youthreport1_cddr94", "cocaine_nr_past_month"),
       ])]

VARS["tabular"]['opiates_initiation'] = [("youthreport1", [
       ("youthreport1_cddr64", "opiates_age_first"),
       ("youthreport1_cddr65", "opiates_age_regular"),
       ("youthreport1_cddr66a", "opiates_nr_lifetime"),
       ("youthreport1_cddr68", "opiates_nr_past_year"),
       ("youthreport1_cddr69", "opiates_nr_past_month"),
       ])]

# These are the same
VARS["tabular"]['neuropsych'] = [("cnp", [("cnp_cpf_ifac_tot", "CPF_total_correct"),
                      ("cnp_cpw_iwrd_tot", "CPW_total_correct"),
                      ("cnp_spcptnl_scpt_tp", "NumLet_true_positives"),
                      ("cnp_pmat24a_pmat24_a_cr", "PMAT24_total_correct"),
                      ("cnp_cpfd_dfac_tot", "CPFD_total_correct"),
                      ("cnp_cpwd_dwrd_tot", "CPWD_total_correct"),
                      ("cnp_shortvolt_svt", "SVOLT_total_correct"),
                      ("cnp_er40d_er40_cr", "ER40_total_correct"),
                      ("cnp_pcet_pcet_acc2", "PECT_accuracy"),
                      ("cnp_medf36_medf36_a", "MET_total_correct"),
                      ("cnp_pvrt_pvrtcr", "SPVRT_total_correct"),
                      ("cnp_svdelay_svt_ld", "SVOLTD_total_correct"), 
                      ])]

# These are different, I did not make four of them binary
VARS["tabular"]['peers'] = [("youthreport2_prr", [("youthreport2_prr1a", "nr_same_sex_friends"),
                               ("youthreport2_prr1b", "nr_opposite_sex_friends"),
                               ("youthreport2_prr6", "nr_dating_partners")]),
                        ("youthreport2_pgd", [("youthreport2_pgd_sec1_pgd2", "nr_friends_drink_alcohol"),
                               ("youthreport2_pgd_sec1_pgd3", "nr_friends_get_drunk"),
                               ("youthreport2_pgd_sec1_pgd4", "nr_friends_problems_alcohol")])]

# These two are the same
VARS["tabular"]['sleep'] = [("clinical", [("shq_sleepiness", "sleepiness"),
                       ("shq_circadian", "circadian_rhythm")])]
                       #("shq_weekday_sleep", "weekday_sleep"),
                       #("shq_weekend_sleep", "weekend_sleep"),
                       #("shq_weekend_bedtime_delay", "weekend_bedtime_delay"),
                       #("shq_weekend_wakeup_delay", "weekend_wakeup_delay"),
                       #("casq_score", "casq_score")])]

# These are the same
VARS["tabular"]['personality'] = [("clinical", [("tipi_agv", "agreeableness"),
                             ("tipi_csv", "conscientiousness"),
                             ("tipi_ems", "emotional_stability"),
                             ("tipi_etv", "extraversion"),
                             ("tipi_ope", "openness_to_experiences"),
                             ("upps_nug", "negative_urgency"),
                             ("upps_pmt", "premeditation"),
                             ("upps_psv", "perseverance"),
                             ("upps_pug", "positive_urgency"),
                             ("upps_sss", "sensation_seeking")])]

# These six are the same
VARS["tabular"]['exec_functioning'] = [("brief", [("brief_inhibit_raw", "brief_inhibit_raw"),
                      ("brief_beh_shift_raw", "brief_beh_shift_raw"),
                      ("brief_control_raw", "brief_control_raw"),       
                      ("brief_plan_raw", "brief_plan_raw"),                      
                      ("brief_task_raw", "brief_task_raw"),                      
                      ("brief_gec_raw", "brief_gec_raw")])]
                      #("brief_cog_shift_raw", "brief_cog_shift_t"),
                      #("brief_monitor_raw", "brief_monitor_t"),
                      #("brief_memory_raw", "brief_memory_t"),
                      #("brief_materials_raw", "brief_materials_t"),
                      #("brief_shift_raw", "brief_shift_t"),
                      #("brief_bri_raw", "brief_bri_t"),
                      #("brief_mi_raw", "brief_mi_t")]
                      
# The same
VARS["tabular"]['biological'] = [("clinical", [("pds_pubcat", "pubertal_category")])]

# Slightly different
VARS["tabular"]['neighborhood'] = [("youthreport2_aay", [("youthreport2_aay_set1_aay5", "alcohol_use_prevention"),
                                      ("youthreport2_aay_set1_aay2", "level_community_resources"),
                                      ("youthreport2_aay_set2_aay6", "difficulty_alcohol_home"),
                                      ("youthreport2_aay_set2_aay7", "difficulty_alcohol_neighborhood"),
                                      ("youthreport2_aay_set2_aay8", "difficulty_alcohol_outside")])]
                                      # ("youthreport2_aay_set1_aay1", "strong_community_identity"),
                                      # ("youthreport2_aay_set3_aay12", "difficulty_get_drugs_home"),
                                      # ("youthreport2_aay_set3_aay13", "difficulty_get_drugs_neighborhood"),
                                      # ("youthreport2_aay_set3_aay14", "difficulty_get_drugs_outside"),
                                      # ("youthreport2_aay_set3_aay15", "difficulty_get_drugs_anywhere")])]

# The same
VARS["tabular"]['attitudes_beliefs'] = [("clinical", [("aeq_csb", "changes_social_beh"),
                                   ("aeq_gpc", "global_positive_change"),
                                   ("aeq_ia", "increased_arousal"),
                                   ("aeq_icma", "improved_cog_motor_ability"),
                                   ("aeq_rtr", "relaxation_tension_reduction")])]

VARS["tabular"]['ses'] = [("youthreport1_ses", [
       ("youthreport1_ses4", "parent_a_education"), 
       ("youthreport1_ses5", "parent_b_education"), 
       ("youthreport1_ses14", "combined_parental_income")])]

#VARS["tabular"]['mental_health'] = [("ysr", [("ysr_asr_internal_raw", "internalizing_problems"),
#                               ("ysr_asr_external_raw", "externalizing_problems")])]

# Here, I am missing 2 new ones
VARS["structural"]["sMRI"] = [("lh.aparc", [("frontalpole_grayvol", "l_frontalpole_grayvol"),
                            ("superiorfrontal_thickavg", "l_superiorfrontal_thickavg"),
                            ("lateralorbitofrontal_grayvol", "l_lateralorbitofrontal_grayvol"),
                            ("rostralmiddlefrontal_thickavg", "l_rostralmiddlefrontal_thickavg"),
                            ("paracentral_thickavg", "l_paracentral_thickavg"),
                            ("pericalcarine_thickavg", "l_pericalcarine_thickavg"),
                            ("lateraloccipital_thickavg", "l_lateraloccipital_thickavg"),
                            ("temporalpole_grayvol", "l_temporalpole_grayvol"),
                            ("supramarginal_thickavg", "l_supramarginal_thickavg"),
                            ("transversetemporal_thickavg", "l_transversetemporal_thickavg"),
                            ("parsorbitalis_surfarea", "l_parsorbitalis_surfarea")]),
              ("rh.aparc", [("insula_grayvol", "r_insula_grayvol"),
                            ("rostralmiddlefrontal_grayvol", "r_rostralmiddlefrontal_grayvol"),
                            ("supramarginal_thickavg", "r_supramarginal_thickavg"),
                            ("inferiorparietal_thickavg", "r_inferiorparietal_thickavg"),
                            ("pericalcarine_thickavg", "r_pericalcarine_thickavg"),
                            ("cuneus_thickavg", "r_cuneus_thickavg"),
                            ("parsorbitalis_thickavg", "r_parsorbitalis_thickavg"),
                            ("superiorparietal_thickavg", "r_superiorparietal_thickavg"),
                            ("precuneus_thickavg", "r_precuneus_thickavg"),
                            ("temporalpole_thickavg", "r_temporalpole_thickavg"),
                            ("frontalpole_thickavg", "r_frontalpole_thickavg"),
                            ("precentral_grayvol", "r_precentral_grayvol"),
                            ("parahippocampal_surfarea", "r_parahippocampal_surfarea")]),
              ("aseg", [("measure_brainsegvol", "brainsegvol"),
                        ("right_amygdala_volume_mm3", "right_amygdala_volume_mm3"), 
                        ("right_cerebellum_cortex_volume_mm3", "right_cerebellum_cortex_volume_mm3"),
                        ("right_pallidum_volume_mm3", "right_pallidum_volume_mm3"),
                        ("left_accumbens_area_volume_mm3", "left_accumbens_area_volume_mm3")])]

# Load the fMRI correlations
VARS["functional"]["QA"] = [("rs_qa", [("rs_num_outliers", "rs_num_outliers")])]
fMRI_parcels = {"parc116": ["Precentral_L","Precentral_R","Frontal_Sup_L","Frontal_Sup_R","Frontal_Sup_Orb_L","Frontal_Sup_Orb_R","Frontal_Mid_L","Frontal_Mid_R","Frontal_Mid_Orb_L","Frontal_Mid_Orb_R","Frontal_Inf_Oper_L","Frontal_Inf_Oper_R","Frontal_Inf_Tri_L","Frontal_Inf_Tri_R","Frontal_Inf_Orb_L","Frontal_Inf_Orb_R","Rolandic_Oper_L","Rolandic_Oper_R","Supp_Motor_Area_L","Supp_Motor_Area_R","Olfactory_L","Olfactory_R","Frontal_Sup_Medial_L","Frontal_Sup_Medial_R","Frontal_Med_Orb_L","Frontal_Med_Orb_R","Rectus_L","Rectus_R","Insula_L","Insula_R","Cingulum_Ant_L","Cingulum_Ant_R","Cingulum_Mid_L","Cingulum_Mid_R","Cingulum_Post_L","Cingulum_Post_R","Hippocampus_L","Hippocampus_R","ParaHippocampal_L","ParaHippocampal_R","Amygdala_L","Amygdala_R","Calcarine_L","Calcarine_R","Cuneus_L","Cuneus_R","Lingual_L","Lingual_R","Occipital_Sup_L","Occipital_Sup_R","Occipital_Mid_L","Occipital_Mid_R","Occipital_Inf_L","Occipital_Inf_R","Fusiform_L","Fusiform_R","Postcentral_L","Postcentral_R","Parietal_Sup_L","Parietal_Sup_R","Parietal_Inf_L","Parietal_Inf_R","SupraMarginal_L","SupraMarginal_R","Angular_L","Angular_R","Precuneus_L","Precuneus_R","Paracentral_Lobule_L","Paracentral_Lobule_R","Caudate_L","Caudate_R","Putamen_L","Putamen_R","Thalamus_L","Thalamus_R","Heschl_L","Heschl_R","Temporal_Sup_L","Temporal_Sup_R","Temporal_Pole_Sup_L","Temporal_Pole_Sup_R","Temporal_Mid_L","Temporal_Mid_R","Temporal_Pole_Mid_L","Temporal_Pole_Mid_R","Temporal_Inf_L","Temporal_Inf_R","Cerebelum_Crus1_L","Cerebelum_Crus1_R","Cerebelum_Crus2_L","Cerebelum_Crus2_R","Cerebelum_3_L","Cerebelum_3_R","Cerebelum_4_5_L","Cerebelum_4_5_R","Cerebelum_6_L","Cerebelum_6_R","Cerebelum_7b_L","Cerebelum_7b_R","Cerebelum_8_L","Cerebelum_8_R","Cerebelum_9_L","Cerebelum_9_R","Cerebelum_10_L","Cerebelum_10_R","Vermis_1","Vermis_2","Vermis_3"],
                "parc215": ["lh_vis_1","lh_vis_2","lh_vis_3","lh_vis_4","lh_vis_5","lh_vis_6","lh_vis_7","lh_vis_8","lh_vis_9","lh_vis_10","lh_vis_11","lh_vis_12","lh_vis_13","lh_vis_14","lh_sommot_1","lh_sommot_2","lh_sommot_3","lh_sommot_4","lh_sommot_5","lh_sommot_6","lh_sommot_7","lh_sommot_8","lh_sommot_9","lh_sommot_10","lh_sommot_11","lh_sommot_12","lh_sommot_13","lh_sommot_14","lh_sommot_15","lh_sommot_16","lh_dorsattn_post_1","lh_dorsattn_post_2","lh_dorsattn_post_3","lh_dorsattn_post_4","lh_dorsattn_post_5","lh_dorsattn_post_6","lh_dorsattn_post_7","lh_dorsattn_post_8","lh_dorsattn_post_9","lh_dorsattn_post_10","lh_dorsattn_fef_1","lh_dorsattn_fef_2","lh_dorsattn_prcv_1","lh_salventattn_paroper_1","lh_salventattn_paroper_2","lh_salventattn_paroper_3","lh_salventattn_froper_1","lh_salventattn_froper_2","lh_salventattn_froper_3","lh_salventattn_froper_4","lh_salventattn_pfcl_1","lh_salventattn_med_1","lh_salventattn_med_2","lh_salventattn_med_3","lh_limbic_ofc_1","lh_limbic_ofc_2","lh_limbic_temppole_1","lh_limbic_temppole_2","lh_limbic_temppole_3","lh_limbic_temppole_4","lh_cont_par_1","lh_cont_par_2","lh_cont_par_3","lh_cont_temp_1","lh_cont_pfcl_1","lh_cont_pfcl_2","lh_cont_pfcl_3","lh_cont_pfcl_4","lh_cont_pfcl_5","lh_cont_pfcl_6","lh_cont_pcun_1","lh_cont_cing_1","lh_cont_cing_2","lh_default_temp_1","lh_default_temp_2","lh_default_temp_3","lh_default_temp_4","lh_default_temp_5","lh_default_temp_6","lh_default_temp_7","lh_default_temp_8","lh_default_temp_9","lh_default_pfc_1","lh_default_pfc_2","lh_default_pfc_3","lh_default_pfc_4","lh_default_pfc_5","lh_default_pfc_6","lh_default_pfc_7","lh_default_pfc_8","lh_default_pfc_9","lh_default_pfc_10","lh_default_pfc_11","lh_default_pfc_12","lh_default_pfc_13","lh_default_pcc_1","lh_default_pcc_2","lh_default_pcc_3","lh_default_pcc_4","lh_default_phc_1","rh_vis_1","rh_vis_2","rh_vis_3","rh_vis_4","rh_vis_5","rh_vis_6","rh_vis_7","rh_vis_8","rh_vis_9","rh_vis_10","rh_vis_11","rh_vis_12","rh_vis_13","rh_vis_14","rh_vis_15","rh_sommot_1","rh_sommot_2","rh_sommot_3","rh_sommot_4","rh_sommot_5","rh_sommot_6","rh_sommot_7","rh_sommot_8","rh_sommot_9","rh_sommot_10","rh_sommot_11","rh_sommot_12","rh_sommot_13","rh_sommot_14","rh_sommot_15","rh_sommot_16","rh_sommot_17","rh_sommot_18","rh_sommot_19","rh_dorsattn_post_1","rh_dorsattn_post_2","rh_dorsattn_post_3","rh_dorsattn_post_4","rh_dorsattn_post_5","rh_dorsattn_post_6","rh_dorsattn_post_7","rh_dorsattn_post_8","rh_dorsattn_post_9","rh_dorsattn_post_10","rh_dorsattn_fef_1","rh_dorsattn_fef_2","rh_dorsattn_prcv_1","rh_salventattn_tempoccpar_1","rh_salventattn_tempoccpar_2","rh_salventattn_tempoccpar_3","rh_salventattn_prc_1","rh_salventattn_froper_1","rh_salventattn_froper_2","rh_salventattn_froper_3","rh_salventattn_froper_4","rh_salventattn_med_1","rh_salventattn_med_2","rh_salventattn_med_3","rh_limbic_ofc_1","rh_limbic_ofc_2","rh_limbic_ofc_3","rh_limbic_temppole_1","rh_limbic_temppole_2","rh_limbic_temppole_3","rh_cont_par_1","rh_cont_par_2","rh_cont_par_3","rh_cont_temp_1","rh_cont_pfcv_1","rh_cont_pfcl_1","rh_cont_pfcl_2","rh_cont_pfcl_3","rh_cont_pfcl_4","rh_cont_pfcl_5","rh_cont_pfcl_6","rh_cont_pfcl_7","rh_cont_pcun_1","rh_cont_pfcmp_1","rh_cont_pfcmp_2","rh_cont_pfcmp_3","rh_cont_pfcmp_4","rh_default_par_1","rh_default_par_2","rh_default_par_3","rh_default_temp_1","rh_default_temp_2","rh_default_temp_3","rh_default_temp_4","rh_default_temp_5","rh_default_pfcv_1","rh_default_pfcm_1","rh_default_pfcm_2","rh_default_pfcm_3","rh_default_pfcm_4","rh_default_pfcm_5","rh_default_pfcm_6","rh_default_pfcm_7","rh_default_pcc_1","rh_default_pcc_2","rh_default_pcc_3","lh_thalamus","lh_caudate","lh_putamen","lh_pallidum","lh_hippocampus","lh_amygdala","lh_accumbens","rh_thalamus","rh_caudate","rh_putamen","rh_pallidum","rh_hippocampus","rh_amygdala","rh_accumbens","vta"]
                }
cors_116 = ["{}-{}".format(n1, n1) for n1 in fMRI_parcels["parc116"]] + ["{}-{}".format(n1, n2) for (n1, n2) in combinations(fMRI_parcels["parc116"], 2)]
#corrs_215 = ["{}-{}".format(n1, n1) for n1 in fMRI_parcels["parc215"]] + ["{}-{}".format(n1, n2) for (n1, n2) in combinations(fMRI_parcels["parc215"], 2)]
# Duplicate name to have old and new name values
VARS["functional"]["fMRI_parc116"] = [("sri24_parc116_gm_roi2roi_correlations", [(x, x) for x in cors_116])]
#VARS["functional"]["fMRI_parc215"] = [("sri24_parc215_gm_roi2roi_correlations", [(x, x) for x in corrs_215])]