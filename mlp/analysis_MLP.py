import ROOT
import os

ROOT.gInterpreter.Declare("""
#include <cmath>
#include "TVector3.h"
#include "edm4hep/ReconstructedParticleData.h"
#include "ROOT/RVec.hxx"

namespace CustomKinematics {
    // Compute the isolation angle (theta_iso) in degrees: minimum angle between any two particles in the two hemispheres
    float get_theta_iso(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& pos_parts,
                        const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& neg_parts) {
        if (pos_parts.empty() || neg_parts.empty()) return -999.0f;
        double min_angle_deg = 180.0; // Assume initially that two particles are perfectly antiparallel
        for (const auto& p1 : pos_parts) { // For each p1 in the positive hemisphere, iterate over all p2 in the other and compute the angle between them    
            TVector3 v1(p1.momentum.x, p1.momentum.y, p1.momentum.z);
            for (const auto& p2 : neg_parts) {
                TVector3 v2(p2.momentum.x, p2.momentum.y, p2.momentum.z);
                double ang = v1.Angle(v2) * (180.0 / M_PI);
                if (ang < min_angle_deg) min_angle_deg = ang; // Update the variable if the angle is smaller than the previous ones
            }
        }
        return (float)min_angle_deg;
    }

    // Compute the acollinearity (theta_acol) between tracks in degrees: how much the event deviates from being perfectly antiparallel
    float get_theta_acol(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& pos_parts,
                         const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& neg_parts) {
        if (pos_parts.empty() || neg_parts.empty()) return -999.0f;
        
        auto max_p_pos = pos_parts[0]; // Store the particle with the highest momentum (leading track) in the positive hemisphere in this variable
        float max_val_pos = 0;
        for (const auto& p : pos_parts) { // Iterate to find which particle has the largest momentum magnitude
            float p_val = std::sqrt(p.momentum.x*p.momentum.x + p.momentum.y*p.momentum.y + p.momentum.z*p.momentum.z);
            if (p_val > max_val_pos) { max_val_pos = p_val; max_p_pos = p; }
        }
        
        auto max_p_neg = neg_parts[0]; // Do the same for the negative hemisphere
        float max_val_neg = 0;
        for (const auto& p : neg_parts) {
            float p_val = std::sqrt(p.momentum.x*p.momentum.x + p.momentum.y*p.momentum.y + p.momentum.z*p.momentum.z);
            if (p_val > max_val_neg) { max_val_neg = p_val; max_p_neg = p; }
        }
        
        // Create 3D vectors with the two leading tracks
        TVector3 v1(max_p_pos.momentum.x, max_p_pos.momentum.y, max_p_pos.momentum.z);
        TVector3 v2(max_p_neg.momentum.x, max_p_neg.momentum.y, max_p_neg.momentum.z);
        
        // Dot product between the two leading tracks.
        double cos_val = - (v1.Dot(v2)) / (v1.Mag() * v2.Mag()); // The minus sign ensures that if they are antiparallel, the angle is 0
        if (cos_val > 1.0) cos_val = 1.0; // Prevent potential floating-point rounding errors (1.0000001)
        if (cos_val < -1.0) cos_val = -1.0;
        return (float)(std::acos(cos_val) * (180.0 / M_PI));
    }

    // Compute Erad, Prad, p1_norm, and p2_norm
    std::vector<float> get_rad_vars(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& rps,
                                    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& pos_parts,
                                    const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& neg_parts,
                                    float E_beam) {
        float P_rad = -999.0f;
        float E_rad = -999.0f;
        float p1_norm = -999.0f;
        float p2_norm = -999.0f;

        if (pos_parts.empty() || neg_parts.empty()) return {P_rad, E_rad, p1_norm, p2_norm};

        auto p1_leading = pos_parts[0]; // Consider the leading tracks for the positive hemisphere
        float max_p1 = 0;
        for (const auto& p : pos_parts) {
            if (p.charge != 0) { // Search for leading tracks only among charged tracks
                float p_val = std::sqrt(p.momentum.x*p.momentum.x + p.momentum.y*p.momentum.y + p.momentum.z*p.momentum.z);
                if (p_val > max_p1) { max_p1 = p_val; p1_leading = p; }
            }
        }

        auto p2_leading = neg_parts[0]; // Do the same for the negative hemisphere
        float max_p2 = 0;
        for (const auto& p : neg_parts) {
            if (p.charge != 0) {
                float p_val = std::sqrt(p.momentum.x*p.momentum.x + p.momentum.y*p.momentum.y + p.momentum.z*p.momentum.z);
                if (p_val > max_p2) { max_p2 = p_val; p2_leading = p; }
            }
        }

        if (max_p1 == 0 || max_p2 == 0) return {P_rad, E_rad, p1_norm, p2_norm};

        TVector3 v1(p1_leading.momentum.x, p1_leading.momentum.y, p1_leading.momentum.z); // Leading track vectors
        TVector3 v2(p2_leading.momentum.x, p2_leading.momentum.y, p2_leading.momentum.z);

        float E1 = 0.0f;
        float E2 = 0.0f;
        
        for (const auto& p : rps) { // Loop over all particles in the event
            TVector3 v_p(p.momentum.x, p.momentum.y, p.momentum.z);
            
            // Filter electrons and photons by selecting particles with mass < 10 MeV (electrons and photons) 
            // This "artificially" selects the energy deposited in the electromagnetic calorimeter (muons do not deposit energy)
            if (p.mass < 0.01f) {
                if (v1.Angle(v_p) * (180.0 / M_PI) <= 30.0) {
                    E1 += p.energy;
                }
                if (v2.Angle(v_p) * (180.0 / M_PI) <= 30.0) {
                    E2 += p.energy;
                }
            }
        }

        P_rad = std::sqrt(max_p1*max_p1 + max_p2*max_p2) / E_beam; // Compute Erad and Prad
        E_rad = std::sqrt(E1*E1 + E2*E2) / E_beam;
        p1_norm = max_p1 / E_beam;
        p2_norm = max_p2 / E_beam;

        return {P_rad, E_rad, p1_norm, p2_norm};
    }
}
""")

# Input and output configuration

# Input datasets to use
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
outputDir = "output_root_files_MLP/"
os.makedirs(outputDir, exist_ok=True)

nCPUS    = 4
runBatch = False

class RDFanalysis(): # Standard structure of FCCAnalyses frameworks
    def analysers(df):  # Accepts the original ROOT tree (df) and iteratively adds new columns for each event
        df2 = (
            df
            .Alias("rps", "ReconstructedParticles") # Rename to rps (shorter)

            .Define("RP_e",      "ReconstructedParticle::get_e(rps)") # Extract energy, momentum, and charge for each event
            .Define("RP_px",     "ReconstructedParticle::get_px(rps)")
            .Define("RP_py",     "ReconstructedParticle::get_py(rps)")
            .Define("RP_pz",     "ReconstructedParticle::get_pz(rps)")
            .Define("RP_charge", "ReconstructedParticle::get_charge(rps)")
            
            .Define("E_beam", "45.6f") # Set beam and center-of-mass energy
            .Define("sqrts",  "91.2f") 
            
            .Define("rps_charged", "rps[RP_charge != 0]") # Array containing only charged particles
            .Define("N_ch",        "(int)rps_charged.size()") # Count the length of the charged array to get the charged track multiplicity
            
            .Define("E_tot_norm", "(float)(ROOT::VecOps::Sum(RP_e) / sqrts)") # Normalized total visible energy
            # Compute the transverse momentum of the entire event by summing momenta in the xy plane (transverse to z=beam axis)
            .Define("pT_tot",     "(float)sqrt(ROOT::VecOps::Sum(RP_px)*ROOT::VecOps::Sum(RP_px) + ROOT::VecOps::Sum(RP_py)*ROOT::VecOps::Sum(RP_py))")

            # Find the thrust (maximize the longitudinal projection of momenta) 
            .Define("EVT_thrust0",     'Algorithms::minimize_thrust("Minuit2","Migrad")(RP_px, RP_py, RP_pz)') 
            .Define("Thrust_val",      "(float)EVT_thrust0[0]")
            .Define("cos_theta_thrust", "(float)std::abs(EVT_thrust0[3])")

            # Project each particle onto the thrust axis. 
            .Define("RP_thrustangle0", 'Algorithms::getAxisCosTheta(EVT_thrust0, RP_px, RP_py, RP_pz)')

            # The sign of the cosine defines which hemisphere the particle belongs to
            .Define("rps_pos", "rps[RP_thrustangle0 > 0.0]")
            .Define("rps_neg", "rps[RP_thrustangle0 < 0.0]")
            
            .Define("rps_pos_charged", "rps_pos[ReconstructedParticle::get_charge(rps_pos) != 0]")
            .Define("rps_neg_charged", "rps_neg[ReconstructedParticle::get_charge(rps_neg) != 0]")
            
            # Radiative variables and momenta (4-element array). Call the function defined above
            .Define("rad_vars", "CustomKinematics::get_rad_vars(rps, rps_pos, rps_neg, E_beam)")
            .Define("P_rad", "(float)rad_vars[0]")
            .Define("E_rad", "(float)rad_vars[1]")
            .Define("p1_norm", "(float)rad_vars[2]")
            .Define("p2_norm", "(float)rad_vars[3]")
            
            # Call the functions defined above for the angles as well
            .Define("theta_acol", "CustomKinematics::get_theta_acol(rps_pos_charged, rps_neg_charged)")
            .Define("theta_iso",  "CustomKinematics::get_theta_iso(rps_pos_charged, rps_neg_charged)")
        )
        return df2

    def output():
        return [
            "N_ch",
            "E_tot_norm",
            "pT_tot",
            "Thrust_val",
            "cos_theta_thrust",
            "p1_norm",
            "p2_norm",
            "P_rad",
            "E_rad",
            "theta_acol",
            "theta_iso"
        ]
