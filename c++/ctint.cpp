#include "ctint.hpp"
#include <triqs/mc_tools.hpp>
#include <triqs/det_manip.hpp>
#include <boost/serialization/complex.hpp>

// --------------- The QMC configuration ----------------

// Argument type of g0bar
struct arg_t {
  double tau; // The imaginary time
  int s;      // The auxiliary spin
  int k;
  bool head;
  dcomplex T;
};

// The function that appears in the calculation of the determinant
struct g0bar_tau {
  gf<imtime> const &gt;
  double beta, delta;
  int s;

  dcomplex operator()(arg_t const &x, arg_t const &y) const {
    if ((x.tau == y.tau)) { // G_\sigma(0^-) - \alpha(\sigma s)
      return 1.0 + gt[0](0, 0) - (0.5 + (2 * x.s - 1) * delta);
      // return 1.0 + gt[0](0, 0) - (0.5 + (2 * x.s - 1) * delta);
    }
    auto x_y = x.tau - y.tau;
    bool b   = (x_y >= 0);
    if (!b) x_y += beta;
    dcomplex res = gt[closest_mesh_pt(x_y)](0, 0);
    return (b ? res : -res); // take into account antiperiodicity
  }
};

// The Monte Carlo configuration
struct configuration {
  std::vector<triqs::det_manip::det_manip<g0bar_tau>> Mmatrices;
  int last_change=0;
  int perturbation_order() const { return (Mmatrices[up].size()+Mmatrices[down].size())/2; }
  configuration(block_gf<imtime> &g0tilde_tau, double beta, double delta) {
    for (auto spin : {up, down}) Mmatrices.emplace_back(g0bar_tau{g0tilde_tau[spin], beta, delta, spin}, 100);
  }
};


// ------------ QMC move : inserting a vertex ------------------

struct move_insert {
  configuration *config;
  triqs::mc_tools::random_generator &rng;
  double beta,U,alpha,PU;
  gf<imtime> const &phonon_tau;

  dcomplex attempt() { // Insert an interaction vertex at time tau with aux spin s
    auto k = config->perturbation_order();
    int s     = rng(2);
    double b  = rng(2);
    double t1,t2;
    dcomplex T;
    int s1,s2;
    if (b<PU){
      s1    = rng(2);
      s2    = 1-s1;
      t1 = rng(beta);
      t2 = t1;
      T  = -4 * U * beta / (4 * PU + (1 - PU) * phonon_tau[0](0,0));
    }else{
      s1    = rng(2);
      s2    = rng(2);
      t1    = rng(beta);
      double dt;
      while (true){
        t2 = rng(beta);
        dt = t1 - t2;
        if (dt < 0) dt += beta;
        double Pmax = phonon_tau[0](0,0).real();
        double P    = phonon_tau[closest_mesh_pt(dt)](0,0).real();
        if (rng(Pmax)<P) break;
      }

      T  = 2 * alpha * beta / (4 * PU / phonon_tau[closest_mesh_pt(dt)](0,0) + (1 - PU));
    }

    if (s1==s2){
      int p1=config->Mmatrices[s1].size();
      auto det_ratio = config->Mmatrices[s1].try_insert2(p1,p1+1,p1,p1+1,{t1,s,k,true,T},{t2,s,k,false,T},{t1,s,k,true,T},{t2,s,k,false,T});
      return T * det_ratio/(k+1); // The Metropolis ratio
    }else{
      int p1=config->Mmatrices[s1].size();
      int p2=config->Mmatrices[s2].size();
      auto det_ratio = config->Mmatrices[s1].try_insert(p1,p1,{t1,s,k,true,T},{t1,s,k,true,T});
      det_ratio *= config->Mmatrices[s2].try_insert(p2,p2,{t2,s,k,false,T},{t2,s,k,false,T});
      return T * det_ratio/(k+1); // The Metropolis ratio
    }
    return 0;
  }
  dcomplex accept() {
    for (auto &d : config->Mmatrices) d.complete_operation(); // Finish insertion
    return 1.0;
  }

  void reject() {
    for (auto &d : config->Mmatrices) d.reject_last_try(); // Finish insertion
  }
};


// ------------ QMC move : deleting a vertex ------------------

struct move_remove {
  configuration *config;
  triqs::mc_tools::random_generator &rng;
  double beta,U,alpha,PU;
  gf<imtime> const &phonon_tau;

  dcomplex attempt() {
    auto k = config->perturbation_order();
    if (k <= 0) return 0;    // Config is empty, trying to remove makes no sense

    arg_t vertex;
    double t1,t2;
    dcomplex T;
    int s1,s2,k1,k2;
    int p  = rng(k);
    config->last_change=p;

    for(int spin=0;spin<2;spin++){
      for(int i=0;i<config->Mmatrices[spin].size();i++){
        vertex=config->Mmatrices[spin].get_x(i);
        if(vertex.k==p){
          if(vertex.head){
            t1=vertex.tau;s1=spin;k1=i;
            T=vertex.T;
          }else{
            t2=vertex.tau;s2=spin;k2=i;
          }
        }
      }
    }

    if (s1==s2){
      auto det_ratio = config->Mmatrices[s1].try_remove2(k1,k2,k1,k2);
      return k/T*det_ratio;
    }else{
      auto det_ratio = config->Mmatrices[s1].try_remove(k1,k1);
      det_ratio     *= config->Mmatrices[s2].try_remove(k2,k2);
      return k / T * det_ratio;
    }
  }

  dcomplex accept() {
    for (auto &d : config->Mmatrices) d.complete_operation();
    arg_t vertex;
    for(int spin=0;spin<2;spin++){
      for(int i=0;i<config->Mmatrices[spin].size();i++){
        vertex=config->Mmatrices[spin].get_x(i);
        if(vertex.k>config->last_change){
          vertex.k--;config->Mmatrices[spin].change_one_row_and_one_col(i,i, vertex, vertex);
        }
      }
    }
    return 1.0;
  }

  void reject() {
    for (auto &d : config->Mmatrices) d.reject_last_try(); // Finish insertion
  }                                                        // Nothing to do
};

//  -------------- QMC measurement ----------------

struct measure_M {

  configuration const *config; // Pointer to the MC configuration
  block_gf<imfreq> &Mw;        // reference to M-matrix
  double beta;
  dcomplex Z = 0;
  long count = 0;

  measure_M(configuration const *config_, block_gf<imfreq> &Mw_, double beta_) : config(config_), Mw(Mw_), beta(beta_) { Mw() = 0; }

  void accumulate(dcomplex sign) {
    Z += sign;
    count++;

    for (auto spin : {up, down}) {

      // A lambda to measure the M-matrix in frequency
      auto lambda = [this, spin, sign](arg_t const &x, arg_t const &y, dcomplex M) {
        auto const &mesh = this->Mw[spin].mesh();
        auto phase_step  = -1_j * M_PI * (x.tau - y.tau) / beta;
        auto coeff       = std::exp((2 * mesh.first_index() + 1) * phase_step);
        auto fact        = std::exp(2 * phase_step);
        for (auto const &om : mesh) {
          this->Mw[spin][om](0, 0) += sign * M * coeff;
          coeff *= fact;
        }
      };

      foreach (config->Mmatrices[spin], lambda);
    }
  }

  void collect_results(triqs::mpi::communicator const &c) {
    Mw = triqs::mpi::mpi_all_reduce(Mw, c);
    Z  = triqs::mpi::mpi_all_reduce(Z, c);
    Mw = Mw / (-Z * beta);

    // Print the sign
    if (c.rank() == 0) std::cerr << "Average sign " << Z / c.size() / count << std::endl;
  }
};


// ------------ The main class of the solver ------------------------

ctint_solver::ctint_solver(double beta_, int n_iw, int n_tau) : beta(beta_) {

  g0_iw       = make_block_gf({"up", "down"}, gf<imfreq>{{beta, Fermion, n_iw}, {1, 1}});
  g0tilde_tau = make_block_gf({"up", "down"}, gf<imtime>{{beta, Fermion, n_tau}, {1, 1}});
  phonon_iw   = gf<imfreq>{{beta,Boson,n_iw+1},{1,1}};
  phonon_tau  = gf<imtime>{{beta,Boson,n_tau},{1,1}};
  g0tilde_iw  = g0_iw;
  g_iw        = g0_iw;
  M_iw        = g0_iw;
}

// The method that runs the qmc
void ctint_solver::solve(double U, double alpha, double PU, double delta, int n_cycles, int length_cycle, int n_warmup_cycles, std::string random_name, int max_time) {

  mpi::communicator world;
  triqs::clef::placeholder<0> spin_;
  triqs::clef::placeholder<1> om_;
  phonon_tau()=triqs::gfs::fourier(phonon_iw);
  for (auto spin : {up, down}) { // Apply shift to g0_iw and Fourier transform
    g0tilde_iw[spin](om_) << g0_iw[spin](om_);
    array<dcomplex, 3> mom{{{0}}, {{1}}}; // Fix the moments: 0 + 1/omega
    g0tilde_tau()[spin] = triqs::gfs::fourier(g0tilde_iw[spin],make_const_view(mom));
  }

  // Rank-specific variables
  int verbosity   = (world.rank() == 0 ? 3 : 0);
  int random_seed = 34788 + 928374 * world.rank();

  // Construct a Monte Carlo loop
  triqs::mc_tools::mc_generic<dcomplex> CTQMC(random_name, random_seed, 1.0, verbosity);

  // Prepare the configuration
  auto config = configuration{g0tilde_tau, beta, delta};

  // Register moves and measurements
  CTQMC.add_move(move_insert{&config, CTQMC.get_rng(), beta,U,alpha,PU, phonon_tau}, "insertion");
  CTQMC.add_move(move_remove{&config, CTQMC.get_rng(), beta,U,alpha,PU, phonon_tau}, "removal");
  CTQMC.add_measure(measure_M{&config, M_iw, beta}, "M measurement");

  // Run and collect results
  CTQMC.warmup_and_accumulate(n_warmup_cycles, n_cycles, length_cycle, triqs::utility::clock_callback(max_time));
  CTQMC.collect_results(world);

  // Compute the Green function from Mw
  g_iw[spin_](om_) << g0tilde_iw[spin_](om_)  + g0tilde_iw[spin_](om_) * M_iw[spin_](om_) * g0tilde_iw[spin_](om_);

  // // Set the tail of g_iw to 1/w
  triqs::arrays::array<dcomplex, 3> mom{{{0}}, {{1}}}; // 0 + 1/omega
  for (auto &g : g_iw) replace_by_tail_in_fit_window(g(), make_const_view(mom));
}
