// Full charged-B event record for a generator-level B -> K nu nubar study.
// New driver code: MIT, as in the parent RapidSim repository.
// EvtGen and the ported Belle II model retain their own licences.
#include "EvtKnunu.h"
#include "EvtGen/EvtGen.hh"
#include "EvtGenBase/EvtDecayBase.hh"
#include "EvtGenBase/EvtDecayTable.hh"
#include "EvtGenBase/EvtMTRandomEngine.hh"
#include "EvtGenBase/EvtPDL.hh"
#include "EvtGenBase/EvtParticle.hh"
#include "EvtGenBase/EvtParticleFactory.hh"
#include "EvtGenBase/EvtRadCorr.hh"
#include "EvtGenBase/EvtRandom.hh"
#include "EvtGenExternal/EvtExternalGenFactory.hh"
#include "EvtGenExternal/EvtPHOTOS.hh"
#include "EvtGenExternal/EvtPythia.hh"
#include "TFile.h"
#include "TObjString.h"
#include "TTree.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {
class GuardedRandom : public EvtMTRandomEngine {
public:
    explicit GuardedRandom(unsigned int seed) : EvtMTRandomEngine(seed) {}
    void newEvent(long long i) { event_ = i; draws_ = 0; }
    double random() override {
        if (++draws_ > 1000000)
            throw std::runtime_error("Event " + std::to_string(event_) +
                " exceeded one million random draws; event not skipped");
        return EvtMTRandomEngine::random();
    }
private:
    long long event_ = -1, draws_ = 0;
};

// Pythia rebuilds daughters and resets the parent's EvtGen channel to zero.
// Audit the actual invocation instead of interpreting that reset as an
// exclusive mode. The selected channel need not be Pythia's final channel.
class AuditedPythia : public EvtPythia {
public:
    EvtDecayBase* clone() override { return new AuditedPythia(); }
    void decay(EvtParticle* p) override {
        const int selected = p->getChannel();
        EvtPythia::decay(p);
        p->setAttribute("bpair_pythia_used", 1);
        p->setAttribute("bpair_selected_channel_plus_one", selected + 1);
    }
};

struct Options {
    long long events = 10000;
    unsigned int seed = 1776;
    bool legacyPythiaCodes = false;
    double sqrts = 10.58;
    fs::path output = "bpair.root";
    fs::path data = BPAIR_DEFAULT_DATA_DIR;
    fs::path user = BPAIR_DEFAULT_USER_DECAY;
};

std::string readText(const fs::path& name) {
    std::ifstream file(name);
    if (!file) throw std::runtime_error("Cannot read " + name.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void rejectTauola(const std::string& text, const std::string& name) {
    std::istringstream lines(text);
    const std::regex token("\\bTAUOLA\\b");
    for (std::string line; std::getline(lines, line);) {
        line = line.substr(0, line.find('#'));
        if (std::regex_search(line, token))
            throw std::runtime_error(name + " requests TAUOLA, disabled in this driver. "
                "Use the EvtGen-shipped table with built-in tau models.");
    }
}

Options parse(int argc, char** argv) {
    Options o;
    if (o.data.empty()) {
        const char* env = std::getenv("EVTGEN_ROOT");
        if (env) o.data = env;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout << "BPairGenerator.exe [--events N] [--seed N] [--sqrt-s GeV] "
                "[--output FILE.root] [--data-dir DIR] [--user-decay FILE] "
                "[--pythia-codes 8|6]\nCM frame, fixed energy, charged B pair, "
                "KNUNU signal, generic tag; Pythia on, PHOTOS FSR off, TAUOLA off.\n";
            std::exit(0);
        }
        if (++i >= argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[i];
        std::size_t used = 0;
        if (key == "--events") {
            o.events = std::stoll(value, &used);
            if (used != value.size() || o.events <= 0)
                throw std::runtime_error("--events must be a positive integer");
        } else if (key == "--seed") {
            const auto seed = std::stoll(value, &used);
            if (used != value.size() || seed < 0 ||
                seed > std::numeric_limits<unsigned int>::max())
                throw std::runtime_error("--seed must fit an unsigned 32-bit integer");
            o.seed = static_cast<unsigned int>(seed);
        } else if (key == "--sqrt-s") {
            o.sqrts = std::stod(value, &used);
            if (used != value.size() || !std::isfinite(o.sqrts) || o.sqrts <= 0)
                throw std::runtime_error("Invalid --sqrt-s");
        } else if (key == "--output") o.output = value;
        else if (key == "--data-dir") o.data = value;
        else if (key == "--user-decay") o.user = value;
        else if (key == "--pythia-codes") {
            if (value != "8" && value != "6")
                throw std::runtime_error("--pythia-codes must be 8 or 6");
            o.legacyPythiaCodes = value == "6";
        } else throw std::runtime_error("Unknown option " + key);
    }
    if (fs::exists(o.output) || fs::exists(o.output.string() + ".partial"))
        throw std::runtime_error("Output or .partial file exists; choose a new --output path");
    if (o.output.has_parent_path()) fs::create_directories(o.output.parent_path());
    return o;
}

struct Record {
    Long64_t event = 0;
    int signal = -1, tag = -1, kaon = -1;
    double q2 = 0, cosThetaB = 0;
    std::vector<int> pdg, mother, side, ndaug, charge3, channel;
    std::vector<double> e, px, py, pz, ct, x, y, z, ctau;
    std::vector<std::string> model;
    void bind(TTree& t) {
        t.Branch("event", &event);
        t.Branch("signal_index", &signal); t.Branch("tag_index", &tag);
        t.Branch("kaon_index", &kaon); t.Branch("q2_true", &q2);
        t.Branch("cos_theta_B", &cosThetaB);
        t.Branch("pdg", &pdg); t.Branch("mother", &mother);
        t.Branch("side", &side); t.Branch("n_daughters", &ndaug);
        t.Branch("charge3", &charge3); t.Branch("channel", &channel);
        t.Branch("model", &model);
        t.Branch("E_cm", &e); t.Branch("px_cm", &px);
        t.Branch("py_cm", &py); t.Branch("pz_cm", &pz);
        t.Branch("ct_mm", &ct); t.Branch("x_mm", &x);
        t.Branch("y_mm", &y); t.Branch("z_mm", &z);
        t.Branch("ctau_mm", &ctau);
    }
    int append(EvtParticle* p, int parent, int whichSide) {
        const int index = static_cast<int>(pdg.size());
        const int id = EvtPDL::getStdHep(p->getId());
        if (parent == 0) {
            if (id == 521) { whichSide = 1; signal = index; }
            else if (id == -521) { whichSide = 2; tag = index; }
            else throw std::runtime_error("Upsilon has an unexpected direct daughter");
        }
        if (parent == signal && id == 321) kaon = index;
        pdg.push_back(id); mother.push_back(parent); side.push_back(whichSide);
        ndaug.push_back(static_cast<int>(p->getNDaug()));
        charge3.push_back(EvtPDL::chg3(p->getId()));
        const bool usedPythia = p->getAttribute("bpair_pythia_used") != 0;
        channel.push_back(usedPythia ? p->getAttribute("bpair_selected_channel_plus_one") - 1
                                    : p->getChannel());
        auto* table = EvtDecayTable::getInstance();
        std::string name;
        if (usedPythia) name = "PYTHIA";
        else if (p->getNDaug() > 0 && p->getChannel() >= 0 &&
                 p->getChannel() < table->getNModes(p->getId())) {
            auto* decay = table->findDecayModel(p->getId(), p->getChannel());
            if (decay) name = decay->getName();
        }
        model.push_back(name);
        const auto p4 = p->getP4Lab(); // Root is at rest: EvtGen "lab" = CM.
        e.push_back(p4.get(0)); px.push_back(p4.get(1));
        py.push_back(p4.get(2)); pz.push_back(p4.get(3));
        const auto position = p->get4Pos();
        ct.push_back(position.get(0)); x.push_back(position.get(1));
        y.push_back(position.get(2)); z.push_back(position.get(3));
        ctau.push_back(EvtPDL::getctau(p->getId()));
        for (std::size_t i = 0; i < p->getNDaug(); ++i)
            append(p->getDaug(static_cast<int>(i)), index, whichSide);
        return index;
    }
    void finish() {
        if (signal < 0 || tag < 0 || kaon < 0 || ndaug[signal] != 3)
            throw std::runtime_error("Forced signal topology was not generated");
        double en = 0, xn = 0, yn = 0, zn = 0;
        int neutrinos = 0;
        for (std::size_t i = 0; i < pdg.size(); ++i) {
            const int absid = std::abs(pdg[i]);
            if (mother[i] == signal && (absid == 12 || absid == 14 || absid == 16)) {
                en += e[i]; xn += px[i]; yn += py[i]; zn += pz[i]; ++neutrinos;
            }
            if (ndaug[i] == 0 && ((absid >= 1 && absid <= 8) || absid == 21))
                throw std::runtime_error("Unhadronised terminal parton in output");
        }
        if (neutrinos != 2) throw std::runtime_error("Signal neutrino pair missing");
        q2 = en*en - xn*xn - yn*yn - zn*zn;
        const double momentum = std::sqrt(px[signal]*px[signal] + py[signal]*py[signal] +
                                          pz[signal]*pz[signal]);
        cosThetaB = pz[signal] / momentum;
    }
};
struct DeleteTree { void operator()(EvtParticle* p) const { if (p) p->deleteTree(); } };

int run(const Options& o) {
    const auto generic = readText(o.data / "DECAY.DEC");
    const auto pdl = readText(o.data / "evt.pdl");
    const auto forced = readText(o.user);
    rejectTauola(generic, "Generic decay table"); rejectTauola(forced, "User decay table");
    GuardedRandom random(o.seed);
    EvtRandom::setRandomEngine(&random);
    auto* factory = EvtExternalGenFactory::getInstance();
    const char* pythiaData = std::getenv("PYTHIA8DATA");
    // Current table uses Pythia 8 codes; do not apply legacy conversion to it.
    factory->definePythiaGenerator(pythiaData ? pythiaData : "", o.legacyPythiaCodes, true);
    factory->definePhotosGenerator("gamma", true);
    EvtPHOTOS photos;
    std::list<EvtDecayBase*> extraModels{new AuditedPythia(), new Belle2::EvtKnunu()};
    EvtGen generator((o.data / "DECAY.DEC").string(), (o.data / "evt.pdl").string(),
                     &random, &photos, &extraModels);
    generator.readUDecay(o.user.string());
    EvtRadCorr::setNeverRadCorr(); // Keep a three-body signal; explicit radiative modes remain.
    const auto ups = EvtPDL::getId("Upsilon(4S)");
    const double mb = EvtPDL::getMeanMass(EvtPDL::getId("B+"));
    const double mk = EvtPDL::getMeanMass(EvtPDL::getId("K+"));
    if (o.sqrts <= 2*mb) throw std::runtime_error("sqrt(s) must exceed 2 mB");
    const fs::path partial = o.output.string() + ".partial";
    TFile output(partial.c_str(), "CREATE");
    if (output.IsZombie()) throw std::runtime_error("Cannot create output ROOT file");
    TTree tree("Events", "Upsilon(4S): KNUNU signal B+ and generic B-, CM frame");
    Record record;
    record.bind(tree);
    std::ostringstream config;
    config.precision(17);
    config << "{\"schema_version\":1,\"events\":" << o.events << ",\"seed\":" << o.seed
           << ",\"sqrt_s_GeV\":" << o.sqrts << ",\"mB_GeV\":" << mb << ",\"mK_GeV\":" << mk
           << ",\"frame\":\"ee_CM\",\"signal_pdg\":521,\"tag_pdg\":-521,"
              "\"pythia\":true,\"photos_fsr\":false,\"tauola\":false,"
              "\"beam_spread\":false,\"ISR\":false,\"detector_response\":false,"
              "\"conditional_signal_sample\":true,\"generator\":\"EvtGen\","
           << "\"pythia_mode_code_version\":" << (o.legacyPythiaCodes ? 6 : 8) << "}";
    TObjString(config.str().c_str()).Write("ConfigJSON");
    TObjString(generic.c_str()).Write("GenericDecayTable");
    TObjString(pdl.c_str()).Write("ParticleTable");
    TObjString(forced.c_str()).Write("ForcedDecayTable");
    long long pythiaNodes = 0, tagNeutrinoEvents = 0;
    for (long long i = 0; i < o.events; ++i) {
        record = Record{}; // Member addresses bound to the tree remain unchanged.
        record.event = i;
        const EvtVector4R initial(o.sqrts, 0., 0., 0.);
        std::unique_ptr<EvtParticle, DeleteTree> root(
            EvtParticleFactory::particleFactory(ups, initial));
        root->setVectorSpinDensity();
        random.newEvent(i);
        try { generator.generateDecay(root.get()); }
        catch (...) { root->printTree(); throw; }
        record.append(root.get(), -1, 0);
        record.finish();
        bool tagHasNeutrino = false;
        for (std::size_t j = 0; j < record.pdg.size(); ++j) {
            if (record.model[j] == "PYTHIA") ++pythiaNodes;
            const int id = std::abs(record.pdg[j]);
            if (record.side[j] == 2 && record.ndaug[j] == 0 &&
                (id == 12 || id == 14 || id == 16)) tagHasNeutrino = true;
        }
        tagNeutrinoEvents += tagHasNeutrino;
        if (tree.Fill() < 0) throw std::runtime_error("ROOT tree write failed");
        if ((i+1) % 1000 == 0) std::cout << "BPair: " << i+1 << "/" << o.events << std::endl;
    }
    tree.Write(); tree.SetDirectory(nullptr); output.Close();
    fs::rename(partial, o.output);
    std::cout << "BPair: wrote " << o.events << " events to " << o.output << "\n"
              << "BPair: PYTHIA decay nodes = " << pythiaNodes << "\n"
              << "BPair: events with tag-side neutrinos = " << tagNeutrinoEvents << std::endl;
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return run(parse(argc, argv)); }
    catch (const std::exception& e) {
        std::cerr << "BPairGenerator: " << e.what() << std::endl;
        return 1;
    }
}
