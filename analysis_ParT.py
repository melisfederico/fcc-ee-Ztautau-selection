import ROOT
import os

processList = {
    'p8_ee_Ztautau_ecm91': {'fraction': 0.001, 'chunks': 4},
    'p8_ee_Zee_ecm91':     {'fraction': 0.001, 'chunks': 4},
    'p8_ee_Zmumu_ecm91':   {'fraction': 0.001, 'chunks': 4},
    'p8_ee_Zud_ecm91':     {'fraction': 0.0002, 'chunks': 4},
    'p8_ee_Zss_ecm91':     {'fraction': 0.0002, 'chunks': 4},
    'p8_ee_Zcc_ecm91':     {'fraction': 0.0002, 'chunks': 4},
    'p8_ee_Zbb_ecm91':     {'fraction': 0.0002, 'chunks': 4},
}

procDict  = "FCCee_procDict_winter2023_IDEA.json" # JSON dictionary containing data such as cross-section, number of events

inputDir  = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/"
outputDir = "output_root_files_ParT/"
os.makedirs(outputDir, exist_ok=True)

nCPUS    = 4
runBatch = False

class RDFanalysis(): # Standard structure of FCCAnalyses frameworks
    def analysers(df):  # Accepts the original ROOT tree (df) and iteratively adds new columns for each event
        df2 = (
            df.Alias("rps", "ReconstructedParticles")

            .Define("RP_e",      "ReconstructedParticle::get_e(rps)")
            .Define("RP_px",     "ReconstructedParticle::get_px(rps)")
            .Define("RP_py",     "ReconstructedParticle::get_py(rps)")
            .Define("RP_pz",     "ReconstructedParticle::get_pz(rps)")
            .Define("RP_charge", "ReconstructedParticle::get_charge(rps)")
            
            # Find the thrust (maximize the longitudinal projection of momenta) 
            .Define("EVT_thrust0",      'Algorithms::minimize_thrust("Minuit2","Migrad")(RP_px, RP_py, RP_pz)')
            .Define("cos_theta_thrust", "(float)std::abs(EVT_thrust0[3])")
        )
        return df2

    def output(): # Specify to FCCAnalyses which of the computed event tree columns to save in the output files
        return [
            "RP_e", 
            "RP_px", 
            "RP_py", 
            "RP_pz", 
            "RP_charge",
            "cos_theta_thrust"
        ] 