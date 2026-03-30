/*********************************************************************
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 * 
 *********************************************************************/

/*********************************************************************
 * File: main.c
 * Description: The main routine for the Metric-FastForward Planner.
 *              Modified July 2011 to allow more command-line search 
 *              confiogurations, including improved cost-minimization
 *
 * Author: original version Joerg Hoffmann 2001/2002
 *         modified version Joerg Hoffmann 2012
 * 
 *********************************************************************/ 








#include "ff.h"

#include "memory.h"
#include "output.h"

#include "parse.h"

#include "expressions.h"

#include "inst_pre.h"
#include "inst_easy.h"
#include "inst_hard.h"
#include "inst_final.h"

#include "relax.h"
#include "search.h"











/*
 *  ----------------------------- GLOBAL VARIABLES ----------------------------
 */












/*******************
 * GENERAL HELPERS *
 *******************/








/* used to time the different stages of the planner
 */
float gtempl_time = 0, greach_time = 0, grelev_time = 0, gconn_time = 0;
float gLNF_time = 0, gsearch_time = 0;


/* the command line inputs
 */
struct _command_line gcmd_line;

/* number of states that got heuristically evaluated
 */
int gevaluated_states = 0;

/* maximal depth of breadth first search
 */
int gmax_search_depth = 0;





/***********
 * PARSING *
 ***********/







/* used for pddl parsing, flex only allows global variables
 */
int gbracket_count;
char *gproblem_name;

/* The current input line number
 */
int lineno = 1;

/* The current input filename
 */
char *gact_filename;

/* The pddl domain name
 */
char *gdomain_name = NULL;

/* loaded, uninstantiated operators
 */
PlOperator *gloaded_ops = NULL;

/* stores initials as fact_list 
 */
PlNode *gorig_initial_facts = NULL;

/* not yet preprocessed goal facts
 */
PlNode *gorig_goal_facts = NULL;

/* axioms as in UCPOP before being changed to ops
 */
PlOperator *gloaded_axioms = NULL;

/* the types, as defined in the domain file
 */
TypedList *gparse_types = NULL;

/* the constants, as defined in domain file
 */
TypedList *gparse_constants = NULL;

/* the predicates and their arg types, as defined in the domain file
 */
TypedListList *gparse_predicates = NULL;

/* the functions and their arg types, as defined in the domain file
 */
TypedListList *gparse_functions = NULL;

/* the objects, declared in the problem file
 */
TypedList *gparse_objects = NULL;

/* the metric
 */
Token gparse_optimization;
ParseExpNode *gparse_metric = NULL;


/* connection to instantiation ( except ops, goal, initial )
 */

/* all typed objects 
 */
FactList *gorig_constant_list = NULL;

/* the predicates and their types
 */
FactList *gpredicates_and_types = NULL;

/* the functions and their types
 */
FactList *gfunctions_and_types = NULL;












/*****************
 * INSTANTIATING *
 *****************/









/* global arrays of constant names,
 *               type names (with their constants),
 *               predicate names,
 *               predicate aritys,
 *               defined types of predicate args
 */
Token gconstants[MAX_CONSTANTS];
int gnum_constants = 0;
Token gtype_names[MAX_TYPES];
int gtype_consts[MAX_TYPES][MAX_TYPE];
Bool gis_member[MAX_CONSTANTS][MAX_TYPES];
int gmember_nr[MAX_CONSTANTS][MAX_TYPES];/* nr of object within a type */
int gtype_size[MAX_TYPES];
int gnum_types = 0;
Token gpredicates[MAX_PREDICATES];
int garity[MAX_PREDICATES];
Bool gaxiom_added[MAX_PREDICATES];
int gpredicates_args_type[MAX_PREDICATES][MAX_ARITY];
int gnum_predicates = 0;
Token gfunctions[MAX_FUNCTIONS];
int gf_arity[MAX_FUNCTIONS];
int gfunctions_args_type[MAX_FUNCTIONS][MAX_ARITY];
int gnum_functions = 0;





/* the domain in integer (Fact) representation
 */
Operator_pointer goperators[MAX_OPERATORS];
int gnum_operators = 0;
Fact *gfull_initial;
int gnum_full_initial = 0;
FluentValue *gfull_fluents_initial;
int gnum_full_fluents_initial = 0;
WffNode *ggoal = NULL;

ExpNode *gmetric = NULL;



/* stores inertia - information: is any occurence of the predicate
 * added / deleted in the uninstantiated ops ?
 */
Bool gis_added[MAX_PREDICATES];
Bool gis_deleted[MAX_PREDICATES];


/* for functions we *might* want to say, symmetrically, whether it is
 * increased resp. decreased at all.
 *
 * that is, however, somewhat involved because the right hand
 * sides can be arbirtray expressions, so we have no guarantee
 * that increasing really does adds to a functions value...
 *
 * thus (for the time being), we settle for "is the function changed at all?"
 */
Bool gis_changed[MAX_FUNCTIONS];



/* splitted initial state:
 * initial non static facts,
 * initial static facts, divided into predicates
 * (will be two dimensional array, allocated directly before need)
 */
Facts *ginitial = NULL;
int gnum_initial = 0;
Fact **ginitial_predicate;
int *gnum_initial_predicate;

/* same thing for functions
 */
FluentValues *gf_initial;
int gnum_f_initial = 0;
FluentValue **ginitial_function;
int *gnum_initial_function;



/* the type numbers corresponding to any unary inertia
 */
int gtype_to_predicate[MAX_PREDICATES];
int gpredicate_to_type[MAX_TYPES];

/* (ordered) numbers of types that new type is intersection of
 */
TypeArray gintersected_types[MAX_TYPES];
int gnum_intersected_types[MAX_TYPES];



/* splitted domain: hard n easy ops
 */
Operator_pointer *ghard_operators;
int gnum_hard_operators;
NormOperator_pointer *geasy_operators;
int gnum_easy_operators;



/* so called Templates for easy ops: possible inertia constrained
 * instantiation constants
 */
EasyTemplate *geasy_templates;
int gnum_easy_templates;



/* first step for hard ops: create mixed operators, with conjunctive
 * precondition and arbitrary effects
 */
MixedOperator *ghard_mixed_operators;
int gnum_hard_mixed_operators;



/* hard ''templates'' : pseudo actions
 */
PseudoAction_pointer *ghard_templates;
int gnum_hard_templates;



/* store the final "relevant facts"
 */
Fact grelevant_facts[MAX_RELEVANT_FACTS];
int gnum_relevant_facts = 0;
int gnum_pp_facts = 0;
/* store the "relevant fluents"
 */
Fluent grelevant_fluents[MAX_RELEVANT_FLUENTS];
int gnum_relevant_fluents = 0;
Token grelevant_fluents_name[MAX_RELEVANT_FLUENTS];
/* this is NULL for normal, and the LNF for
 * artificial fluents.
 */
LnfExpNode_pointer grelevant_fluents_lnf[MAX_RELEVANT_FLUENTS];



/* the final actions and problem representation
 */
Action *gactions = NULL;
int gnum_actions;
State ginitial_state;
int *glogic_goal = NULL;
int gnum_logic_goal = 0;
Comparator *gnumeric_goal_comp = NULL;
ExpNode_pointer *gnumeric_goal_lh = NULL, *gnumeric_goal_rh = NULL;
int gnum_numeric_goal = 0;

/* direct numeric goal access
 */
Comparator *gnumeric_goal_direct_comp;
float *gnumeric_goal_direct_c;



/* to avoid memory leaks; too complicated to identify
 * the exact state of the action to throw away (during construction),
 * memory gain not worth the implementation effort.
 */
Action *gtrash_actions = NULL;



/* additional lnf step between finalized inst and
 * conn graph
 */
Comparator *glnf_goal_comp = NULL;
LnfExpNode_pointer *glnf_goal_lh = NULL;
float *glnf_goal_rh = NULL;
int gnum_lnf_goal = 0;

LnfExpNode glnf_metric;
Bool goptimization_established = FALSE;







/**********************
 * CONNECTIVITY GRAPH *
 **********************/







/* one ops (actions) array ...
 */
OpConn *gop_conn;
int gnum_op_conn;



/* one effects array ...
 */
EfConn *gef_conn;
int gnum_ef_conn;



/* one facts array.
 */
FtConn *gft_conn;
int gnum_ft_conn;



/* and: one fluents array.
 */
FlConn *gfl_conn;
int gnum_fl_conn;
int gnum_real_fl_conn;/* number of non-artificial ones */



/* final goal is also transformed one more step.
 */
int *gflogic_goal = NULL;
int gnum_flogic_goal = 0;
Comparator *gfnumeric_goal_comp = NULL;
int *gfnumeric_goal_fl = NULL;
float *gfnumeric_goal_c = NULL;
int gnum_fnumeric_goal = 0;

/* direct access (by relevant fluents)
 */
Comparator *gfnumeric_goal_direct_comp = NULL;
float *gfnumeric_goal_direct_c = NULL;











/*******************
 * SEARCHING NEEDS *
 *******************/











/* applicable actions
 */
int *gA;/* non-axioms */
int gnum_A;
int *gA_axioms; /* axioms */
int gnum_A_axioms;



/* communication from extract 1.P. to search engine:
 * 1P action choice
 */
int *gH;
int gnum_H;
/* added cost of relaxed plan
 */
float gh_cost;
/* hmax value
 */
float ghmax;



/* to store plan
 */
int gplan_ops[MAX_PLAN_LENGTH];
int gnum_plan_ops = 0;



/* stores the states that the current plan goes through
 * ( for knowing where new agenda entry starts from )
 */
State gplan_states[MAX_PLAN_LENGTH + 1];







/* dirty: multiplic. of total-time in final metric LNF
 */
float gtt;







/* the mneed structures
 */
Bool **gassign_influence;
Bool **gTassign_influence;



/* the real var input to the mneed computation.
 */
Bool *gmneed_start_D;
float *gmneed_start_V;



/* does this contain conditional effects?
 * (if it does then the state hashing has to be made more
 *  cautiously)
 */
Bool gconditional_effects;



/* easier to question: are we optimizing or no?
 */
Bool gcost_minimizing;



/* stores current A* weight: this is initially given by user,
 * but changes during anytime search.
 */
float gw;
/* this is the minimum weight, ie we'll stop once the weight update
 * does/would yield a value <= this.
 * if no such minim weight is given, this will be -1
 */
float gmin_w = -1;



/* this one says whether or not we are actually using
 * cost-minimizing rplans.
 * this will be the case by default if we're running cost-
 * minimizing searches. it can be switched off by a flag;
 * it is automatically switched off in case there are
 * numeric preconditions/goals: for this case,
 * cost-minimizing rplans are not implemented (a numeric prec
 * may cause an action to come in "later" on in the RPG although
 * its logical pres are easy. in that case, any new effects will
 * have a smaller RPGcost value than facts we already have waiting.
 * in other words, the "Dijsktra" nature breaks.
 *
 * ... I suppose there may be a generic solution to this that
 * can handle numeric precs/goals. Doesn't seem important enough
 * to bother.
 */
Bool gcost_rplans;













/*
 *  ----------------------------- HEADERS FOR PARSING ----------------------------
 * ( fns defined in the scan-* files )
 */







void get_fct_file_name( char *filename );
void load_ops_file( char *filename );
void load_fct_file( char *filename );











/*
 *  ----------------------------- MAIN ROUTINE ----------------------------
 */





struct tms lstart, lend;





int main( int argc, char *argv[] )

{

  /* resulting name for ops file
   */
  char ops_file[MAX_LENGTH] = "";
  /* same for fct file 
   */
  char fct_file[MAX_LENGTH] = "";
  
  struct tms start, end;

  Bool found_plan;
  int i;
  float cost;

  Bool prev_gcost_rplans;



  times ( &lstart );

  /* command line treatment
   */
  gcmd_line.display_info = 1;
  gcmd_line.debug = 0;

  /* search settings
   */
  gcmd_line.search_config = 5;
  gcmd_line.cost_rplans = TRUE;
  gcmd_line.w = 5;
  gcmd_line.cost_bound = -1;
  
  memset(gcmd_line.ops_file_name, 0, MAX_LENGTH);
  memset(gcmd_line.fct_file_name, 0, MAX_LENGTH);
  memset(gcmd_line.path, 0, MAX_LENGTH);

  if ( argc == 1 || ( argc == 2 && *++argv[0] == '?' ) ) {
    ff_usage();
    exit( 1 );
  }
  if ( !process_command_line( argc, argv ) ) {
    ff_usage();
    exit( 1 );
  }


  /* make file names
   */

  /* one input name missing
   */
  if ( !gcmd_line.ops_file_name || 
       !gcmd_line.fct_file_name ) {
    fprintf(stdout, "\nff: two input files needed\n\n");
    ff_usage();      
    exit( 1 );
  }
  /* add path info, complete file names will be stored in
   * ops_file and fct_file 
   */
  sprintf(ops_file, "%s%s", gcmd_line.path, gcmd_line.ops_file_name);
  sprintf(fct_file, "%s%s", gcmd_line.path, gcmd_line.fct_file_name);


  /* parse the input files
   */

  /* start parse & instantiation timing
   */
  times( &start );
  /* domain file (ops)
   */
  if ( gcmd_line.display_info >= 1 ) {
    printf("\nff: parsing domain file");
  } 
  /* it is important for the pddl language to define the domain before 
   * reading the problem 
   */
  load_ops_file( ops_file );
  /* problem file (facts)
   */  
  if ( gcmd_line.display_info >= 1 ) {
    printf(" ... done.\nff: parsing problem file"); 
  }
  load_fct_file( fct_file );
  if ( gcmd_line.display_info >= 1 ) {
    printf(" ... done.\n\n");
  }

  /* This is needed to get all types.
   */
  build_orig_constant_list();

  /* last step of parsing: see if it's an ADL domain!
   */
  if ( !make_adl_domain() ) {
    printf("\nff: this is not an ADL problem!");
    printf("\n    can't be handled by this version.\n\n");
    exit( 1 );
  }


  /* now instantiate operators;
   */


  /**************************
   * first do PREPROCESSING * 
   **************************/

  /* start by collecting all strings and thereby encoding 
   * the domain in integers.
   */
  encode_domain_in_integers();

  /* inertia preprocessing, first step:
   *   - collect inertia information
   *   - split initial state into
   *        - arrays for individual predicates
   *        - arrays for all static relations
   *        - array containing non - static relations
   */
  do_inertia_preprocessing_step_1();

  /* normalize all PL1 formulae in domain description:
   * (goal, preconds and effect conditions)
   *   - simplify formula
   *   - expand quantifiers
   *   - NOTs down
   */
  normalize_all_wffs();

  /* translate negative preconds: introduce symmetric new predicate
   * NOT-p(..) (e.g., not-in(?ob) in briefcaseworld)
   */
  translate_negative_preconds();

  /* split domain in easy (disjunction of conjunctive preconds)
   * and hard (non DNF preconds) part, to apply 
   * different instantiation algorithms
   */
  split_domain();

  /***********************************************
   * PREPROCESSING FINISHED                      *
   *                                             *
   * NOW MULTIPLY PARAMETERS IN EFFECTIVE MANNER *
   ***********************************************/

  build_easy_action_templates();
  build_hard_action_templates();

  times( &end );
  TIME( gtempl_time );

  times( &start );

  /* perform reachability analysis in terms of relaxed 
   * fixpoint
   */
  perform_reachability_analysis();

  times( &end );
  TIME( greach_time );

  times( &start );

  /* collect the relevant facts and build final domain
   * and problem representations.
   */
  collect_relevant_facts_and_fluents();

  times( &end );
  TIME( grelev_time );


  /* now transform problem to additive normal form,
   * if possible
   */
  times( &start );
  if ( !transform_to_LNF() ) {
    printf("\n\nThis is not a linear task!\n\n");
    exit( 1 );
  }
  times( &end );
  TIME( gLNF_time );
  
  times( &start );

  /* now build globally accessable connectivity graph
   */
  build_connectivity_graph();

  /* now check for acyclic := effects (in expressions.c)
   */
  check_assigncycles();
  /* set the relevanc info (in expressions.c)
   */
  determine_fl_relevance();

  times( &end );
  TIME( gconn_time );

  /***********************************************************
   * we are finally through with preprocessing and can worry *
   * bout finding a plan instead.                            *
   ***********************************************************/

  if ( gcmd_line.display_info ) {
    printf("\n\nff: search configuration is ");
    switch ( gcmd_line.search_config ) {
    case 0:
      printf("Enforced Hill-Climbing, if that fails then best-first search.\nMetric is plan length.");
      printf("\nNO COST MINIMIZATION");
      if ( !gcost_rplans ) {
	printf(" (and no cost-minimizing relaxed plans).");
      } else {
	printf("\nDEBUG ME: cost min rplans in non-cost minimizing search config?\n\n");
	exit( 1 );
      }
      break;
    case 1:
      printf("best-first search.\nMetric is plan length.");
      printf("\nNO COST MINIMIZATION");
      if ( !gcost_rplans ) {
	printf(" (and no cost-minimizing relaxed plans).");
      } else {
	printf("\nDEBUG ME: cost min rplans in non-cost minimizing search config?\n\n");
	exit( 1 );
      }
      break;
    case 2:
      printf("best-first search with helpful actions pruning.\nMetric is plan length.");
      printf("\nNO COST MINIMIZATION.");
      if ( !gcost_rplans ) {
	printf(" (and no cost-minimizing relaxed plans).");
      } else {
	printf("\nDEBUG ME: cost min rplans in non-cost minimizing search config?\n\n");
	exit( 1 );
      }
      break;
    case 3:
      printf("weighted A* with weight %d.", gcmd_line.w);
      if ( goptimization_established ) {
	printf("\nMetric is ");
	print_LnfExpNode( &glnf_metric );
      } else {
	printf(" plan length");
      }
      printf("\nCOST MINIMIZATION DONE");
      if ( !gcost_rplans ) {
	printf(" (WITHOUT cost-minimizing relaxed plans).");
      } else {
	printf(" (WITH cost-minimizing relaxed plans).");
      }
      break;
    case 4:
      printf("A*epsilon with weight %d.", gcmd_line.w);
      if ( goptimization_established ) {
	printf("\nMetric is ");
	print_LnfExpNode( &glnf_metric );
      } else {
	printf("\nError! Optimization criterion not established.\nA*epsilon not defined.\n\n");
	exit( 1 );
      }
      printf("\nCOST MINIMIZATION DONE");
      if ( !gcost_rplans ) {
	printf(" (WITHOUT cost-minimizing relaxed plans).");
      } else {
	printf(" (WITH cost-minimizing relaxed plans).");
      }
      break;
    case 5:
      printf("Enforced Hill-Climbing, then A*epsilon with weight %d.", gcmd_line.w);
      if ( goptimization_established ) {
	printf("\nMetric is ");
	print_LnfExpNode( &glnf_metric );
      } else {
	printf("\nError! Optimization criterion not established.\nA*epsilon not defined.\n\n");
	exit( 1 );
      }
      printf("\nCOST MINIMIZATION DONE");
      if ( !gcost_rplans ) {
	printf(" (WITHOUT cost-minimizing relaxed plans).");
      } else {
	printf(" (WITH cost-minimizing relaxed plans).");
      }
      break;
    default:
      printf("\n\nUnknown search configuration: %d\n\n", gcmd_line.search_config );
      exit( 1 );
    }
  } else {
    if ( gcmd_line.search_config == 4 && !goptimization_established ) {
      exit( 1 );
    }
  }


  times( &start );



  /* need to evaluate derived predicates in initial state!
   */
  do_axiom_update( &ginitial_state );


  if ( !gcost_rplans ) {
    gcmd_line.cost_bound = -1;
  }

  switch ( gcmd_line.search_config ) {
  case 0:
    found_plan = do_enforced_hill_climbing();
    if ( found_plan ) {
      if ( gcmd_line.display_info ) {
	print_plan();
      }
    } else {
      if ( gcmd_line.display_info ) {
	printf("\n\nEnforced Hill-climbing failed !");
	printf("\nswitching to Best-first Search now.\n");
      }
      do_best_first_search();
    }
    break;
  case 1:
  case 2:
    do_best_first_search();
    break;
  case 3:
    do_weighted_Astar();
    break;
  case 4:
    do_Astar_epsilon();
    break;
  case 5:
    /* gcost_rplans controls whether or not we compute cost-minimal relaxed plans
     * gcost_minimizing is only used in h fn to decide whether or not we
     * need to count the weights of the operators in the relaxed plan.
     *
     * gcost_rplans may be false even for search options 3,4,5, namely if there are
     * numeric preconditions/goals which make this relaxed plan variant invalid.
     * hence we need to remember, when switching it off for EHC, whether or not
     * it was previously on.
     */
    prev_gcost_rplans = gcost_rplans;
    gcost_rplans = FALSE;
    gcost_minimizing = FALSE;
    found_plan = do_enforced_hill_climbing();
    if ( found_plan ) {
      print_plan();
    } else {
      if ( gcmd_line.display_info ) {
	printf("\n\nEnforced Hill-climbing not successful.");
	printf("\nSwitching to A*epsilon now.");
      }
      gcost_rplans = prev_gcost_rplans;
      gcost_minimizing = TRUE;
      do_Astar_epsilon();
    }
    break;
  default:
    printf("\n\nUnknown search configuration: %d\n\n", gcmd_line.search_config );
    exit( 1 );
  }

  times( &end );
  TIME( gsearch_time );



  output_planner_info();

  printf("\n\n");
  exit( 0 );

}











/*
 *  ----------------------------- HELPING FUNCTIONS ----------------------------
 */












void output_planner_info( void )

{

  printf( "\n\ntime spent: %7.2f seconds instantiating %d easy, %d hard action templates", 
	  gtempl_time, gnum_easy_templates, gnum_hard_mixed_operators );
  printf( "\n            %7.2f seconds reachability analysis, yielding %d facts and %d actions", 
	  greach_time, gnum_pp_facts, gnum_actions );
  printf( "\n            %7.2f seconds creating final representation with %d relevant facts, %d relevant fluents", 
	  grelev_time, gnum_relevant_facts, gnum_relevant_fluents );
  printf( "\n            %7.2f seconds computing LNF",
	  gLNF_time );
  printf( "\n            %7.2f seconds building connectivity graph",
	  gconn_time );
  printf( "\n            %7.2f seconds searching, evaluating %d states, to a max depth of %d", 
	  gsearch_time, gevaluated_states, gmax_search_depth );
  printf( "\n            %7.2f seconds total time", 
	  gtempl_time + greach_time + grelev_time + gLNF_time + gconn_time + gsearch_time );

  printf("\n\n");

  exit( 0 );

}



void ff_usage( void )

{

  printf("\nusage of ff:\n");

  printf("\nOPTIONS   DESCRIPTIONS\n\n");
  printf("-p <str>    Path for operator and fact file\n");
  printf("-o <str>    Operator file name\n");
  printf("-f <str>    Fact file name\n\n");

  printf("-r <int>    Random seed [used for random restarts; preset: 0]\n\n");

  printf("-s <int>    Search configuration [preset: s=5]; '+H': helpful actions pruning\n");
  printf("      0     Standard-FF: EHC+H then BFS (cost minimization: NO)\n");
  printf("      1     BFS (cost minimization: NO)\n");
  printf("      2     BFS+H (cost minimization: NO)\n");
  printf("      3     Weighted A* (cost minimization: YES)\n");
  printf("      4     A*epsilon (cost minimization: YES)\n");
  printf("      5     EHC+H then A*epsilon (cost minimization: YES)\n");
  printf("-w <num>    Set weight w for search configs 3,4,5 [preset: w=5]\n\n");

  printf("-C          Do NOT use cost-minimizing relaxed plans for options 3,4,5\n\n");

  printf("-b <float>  Fixed upper bound on solution cost (prune based on g+hmax); active only with cost minimization\n\n");

  if ( 0 ) {
    printf("-i <num>    run-time information level( preset: 1 )\n");
    printf("      0     only times\n");
    printf("      1     problem name, planning process infos\n");
    printf("    101     parsed problem data\n");
    printf("    102     cleaned up ADL problem\n");
    printf("    103     collected string tables\n");
    printf("    104     encoded domain\n");
    printf("    105     predicates inertia info\n");
    printf("    106     splitted initial state\n");
    printf("    107     domain with Wff s normalized\n");
    printf("    108     domain with NOT conds translated\n");
    printf("    109     splitted domain\n");
    printf("    110     cleaned up easy domain\n");
    printf("    111     unaries encoded easy domain\n");
    printf("    112     effects multiplied easy domain\n");
    printf("    113     inertia removed easy domain\n");
    printf("    114     easy action templates\n");
    printf("    115     cleaned up hard domain representation\n");
    printf("    116     mixed hard domain representation\n");
    printf("    117     final hard domain representation\n");
    printf("    118     reachability analysis results\n");
    printf("    119     facts selected as relevant\n");
    printf("    120     final domain and problem representations\n");
    printf("    121     normalized expressions representation\n");
    printf("    122     LNF: translated subtractions representation\n");
    printf("    123     summarized effects LNF  representation\n");
    printf("    124     encoded LNF representation\n");
    printf("    125     connectivity graph\n");
    printf("    126     fixpoint result on each evaluated state\n");
    printf("    127     1P extracted on each evaluated state\n");
    printf("    128     H set collected for each evaluated state\n");
    
    printf("\n-d <num>    switch on debugging\n\n");
  }

}



Bool process_command_line( int argc, char *argv[] )

{

  char option;

  while ( --argc && ++argv ) {
    if ( *argv[0] != '-' || strlen(*argv) != 2 ) {
      return FALSE;
    }
    option = *++argv[0];
    switch ( option ) {
/*     case 'E': */
/*       gcmd_line.ehc = FALSE; */
/*       break; */
/*     case 'O': */
/*       gcmd_line.optimize = TRUE; */
/*       gcmd_line.ehc = FALSE; */
/*       break;       */
    case 'C':
      gcmd_line.cost_rplans = FALSE;
      break;
    default:
      if ( --argc && ++argv ) {
	switch ( option ) {
	case 'p':
	  strncpy( gcmd_line.path, *argv, MAX_LENGTH );
	  break;
	case 'o':
	  strncpy( gcmd_line.ops_file_name, *argv, MAX_LENGTH );
	  break;
	case 'f':
	  strncpy( gcmd_line.fct_file_name, *argv, MAX_LENGTH );
	  break;
	case 'i':
	  sscanf( *argv, "%d", &gcmd_line.display_info );
	  break;
	case 'd':
	  sscanf( *argv, "%d", &gcmd_line.debug );
	  break;
	case 's':
	  sscanf( *argv, "%d", &gcmd_line.search_config );
	  break;
	case 'w':
	  sscanf( *argv, "%d", &gcmd_line.w );
	  break;
	case 'b':
	  sscanf( *argv, "%f", &gcmd_line.cost_bound );
	  break;
	default:
	  printf( "\nff: unknown option: %c entered\n\n", option );
	  return FALSE;
	}
      } else {
	return FALSE;
      }
    }
  }

  if ( 0 > gcmd_line.search_config || gcmd_line.search_config > 5 ) {
    printf("\n\nff: unknown search configuration %d.\n\n", 
	   gcmd_line.search_config);
    return FALSE;
  }

  if ( gcmd_line.search_config <= 2 ) {
    gcost_minimizing = FALSE;
    gcost_rplans = FALSE;
  } else {
    gcost_minimizing = TRUE;
    gcost_rplans = TRUE;
  }

  gw = gcmd_line.w;

  if ( !gcmd_line.cost_rplans ) {
    gcost_rplans = FALSE;
  }

  if ( gcmd_line.cost_bound != -1 && gcmd_line.cost_bound < 0 ) {
    printf("\n\nff: invalid cost bound %f; must be >= 0.\n\n", 
	   gcmd_line.cost_bound);
    return FALSE;
  }

  return TRUE;

}

