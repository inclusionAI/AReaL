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
 * File: inst_final.c
 * Description: final domain representation functions
 *
 *
 * Author: Joerg Hoffmann 2000
 *
 *********************************************************************/ 









#include "ff.h"

#include "output.h"
#include "memory.h"

#include "expressions.h"

#include "inst_pre.h"
#include "inst_final.h"














/********************************
 * POSSIBLY TRUE FACTS ANALYSIS *
 ********************************/








/* local globals for this part
 */

int_pointer lpos[MAX_PREDICATES];
int_pointer lneg[MAX_PREDICATES];
int_pointer luse[MAX_PREDICATES];
int_pointer lindex[MAX_PREDICATES];

int lp;
int largs[MAX_VARS];



/* for collecting poss. defined fluents
 */
int_pointer lf_def[MAX_FUNCTIONS];
int_pointer lf_index[MAX_FUNCTIONS];

int lf;
int lf_args[MAX_VARS];






void perform_reachability_analysis( void )

{

  int size, i, j, k, adr, num, pargtype;
  Bool fixpoint;
  Facts *f;
  NormOperator *no;
  EasyTemplate *t1, *t2;
  NormEffect *ne;
  Action *tmp, *a;
  Bool *had_hard_template;
  PseudoAction *pa;
  PseudoActionEffect *pae;

  gactions = NULL;
  gnum_actions = 0;

  for ( i = 0; i < gnum_predicates; i++ ) {
    size =  1;
    for ( j = 0; j < garity[i]; j++ ) {
      pargtype = gpredicates_args_type[i][j];
      size *= gtype_size[pargtype];
    }

    lpos[i] = ( int_pointer ) calloc( size, sizeof( int ) );
    lneg[i] = ( int_pointer ) calloc( size, sizeof( int ) );
    luse[i] = ( int_pointer ) calloc( size, sizeof( int ) );
    lindex[i] = ( int_pointer ) calloc( size, sizeof( int ) );

    for ( j = 0; j < size; j++ ) {
      lpos[i][j] = 0;
      lneg[i][j] = 1;/* all facts but initials are poss. negative */
      luse[i][j] = 0;
      lindex[i][j] = -1;
    }
  }

  had_hard_template = ( Bool * ) calloc( gnum_hard_templates, sizeof( Bool ) );
  for ( i = 0; i < gnum_hard_templates; i++ ) {
    had_hard_template[i] = FALSE;
  }

  /* mark initial facts as possibly positive, not poss. negative
   */
  for ( i = 0; i < gnum_predicates; i++ ) {
    lp = i;
    for ( j = 0; j < gnum_initial_predicate[i]; j++ ) {
      for ( k = 0; k < garity[i]; k++ ) {
	largs[k] = ginitial_predicate[i][j].args[k];
      }
      adr = fact_adress();
      lpos[lp][adr] = 1;
      lneg[lp][adr] = 0;
    }
  }

  /* compute fixpoint
   */
  fixpoint = FALSE;
  while ( !fixpoint ) {
    fixpoint = TRUE;

    /* assign next layer of easy templates to possibly positive fixpoint
     */
    t1 = geasy_templates;
    while ( t1 ) {
      no = t1->op;
      for ( i = 0; i < no->num_preconds; i++ ) {
	lp = no->preconds[i].predicate;
	for ( j = 0; j < garity[lp]; j++ ) {
	  largs[j] = ( no->preconds[i].args[j] >= 0 ) ?
	    no->preconds[i].args[j] : t1->inst_table[DECODE_VAR( no->preconds[i].args[j] )];
	}
	if ( !lpos[lp][fact_adress()] ) {
	  break;
	}
      }

      if ( i < no->num_preconds ) {
	t1 = t1->next;
	continue;
      }

      num = 0;
      for ( ne = no->effects; ne; ne = ne->next ) {
	num++;
	/* currently, simply ignore effect conditions and assume
	 * they will all be made true eventually.
	 */
	for ( i = 0; i < ne->num_adds; i++ ) {
	  lp = ne->adds[i].predicate;
	  for ( j = 0; j < garity[lp]; j++ ) {
	    largs[j] = ( ne->adds[i].args[j] >= 0 ) ?
	      ne->adds[i].args[j] : t1->inst_table[DECODE_VAR( ne->adds[i].args[j] )];
	  }
	  adr = fact_adress();
	  if ( !lpos[lp][adr] ) {
	    /* new relevant fact! (added non initial)
	     */
	    lpos[lp][adr] = 1;
	    lneg[lp][adr] = 1;
	    luse[lp][adr] = 1;
	    if ( gnum_relevant_facts == MAX_RELEVANT_FACTS ) {
	      printf("\ntoo many relevant facts! increase MAX_RELEVANT_FACTS (currently %d)\n\n",
		     MAX_RELEVANT_FACTS);
	      exit( 1 );
	    }
	    grelevant_facts[gnum_relevant_facts].predicate = lp;
	    for ( j = 0; j < garity[lp]; j++ ) {
	      grelevant_facts[gnum_relevant_facts].args[j] = largs[j];
	    }
	    lindex[lp][adr] = gnum_relevant_facts;
	    gnum_relevant_facts++;
	    fixpoint = FALSE;
	  }
	}
      }

      tmp = new_Action();
      tmp->norm_operator = no;
      tmp->axiom = no->operator->axiom;
      for ( i = 0; i < no->num_vars; i++ ) {
	tmp->inst_table[i] = t1->inst_table[i];
      }
      tmp->name = no->operator->name;
      tmp->num_name_vars = no->operator->number_of_real_params;
      make_name_inst_table_from_NormOperator( tmp, no, t1 );
      tmp->next = gactions;
      tmp->num_effects = num;
      gactions = tmp;
      gnum_actions++;

      t2 = t1->next;
      if ( t1->next ) {
	t1->next->prev = t1->prev;
      }
      if ( t1->prev ) {
	t1->prev->next = t1->next;
      } else {
	geasy_templates = t1->next;
      }
      free_single_EasyTemplate( t1 );
      t1 = t2;
    }

    /* now assign all hard templates that have not been transformed
     * to actions yet.
     */
    for ( i = 0; i < gnum_hard_templates; i++ ) {
      if ( had_hard_template[i] ) {
	continue;
      }
      pa = ghard_templates[i];

      for ( j = 0; j < pa->num_preconds; j++ ) {
	lp = pa->preconds[j].predicate;
	for ( k = 0; k < garity[lp]; k++ ) {
	  largs[k] = pa->preconds[j].args[k];
	}
	if ( !lpos[lp][fact_adress()] ) {
	  break;
	}
      }

      if ( j < pa->num_preconds ) {
	continue;
      }

      for ( pae = pa->effects; pae; pae = pae->next ) {
	/* currently, simply ignore effect conditions and assume
	 * they will all be made true eventually.
	 */
	for ( j = 0; j < pae->num_adds; j++ ) {
	  lp = pae->adds[j].predicate;
	  for ( k = 0; k < garity[lp]; k++ ) {
	    largs[k] = pae->adds[j].args[k];
	  }
	  adr = fact_adress();
	  if ( !lpos[lp][adr] ) {
	    /* new relevant fact! (added non initial)
	     */
	    lpos[lp][adr] = 1;
	    lneg[lp][adr] = 1;
	    luse[lp][adr] = 1;
	    if ( gnum_relevant_facts == MAX_RELEVANT_FACTS ) {
	      printf("\ntoo many relevant facts! increase MAX_RELEVANT_FACTS (currently %d)\n\n",
		     MAX_RELEVANT_FACTS);
	      exit( 1 );
	    }
	    grelevant_facts[gnum_relevant_facts].predicate = lp;
	    for ( k = 0; k < garity[lp]; k++ ) {
	      grelevant_facts[gnum_relevant_facts].args[k] = largs[k];
	    }
	    lindex[lp][adr] = gnum_relevant_facts;
	    gnum_relevant_facts++;
	    fixpoint = FALSE;
	  }
	}
      }

      tmp = new_Action();
      tmp->pseudo_action = pa;
      tmp->axiom = pa->operator->axiom;
      for ( j = 0; j < pa->operator->num_vars; j++ ) {
	tmp->inst_table[j] = pa->inst_table[j];
      }
      tmp->name = pa->operator->name;
      tmp->num_name_vars = pa->operator->number_of_real_params;
      make_name_inst_table_from_PseudoAction( tmp, pa );
      tmp->next = gactions;
      tmp->num_effects = pa->num_effects;
      gactions = tmp;
      gnum_actions++;

      had_hard_template[i] = TRUE;
    }
  }

  free( had_hard_template );

  gnum_pp_facts = gnum_initial + gnum_relevant_facts;

  if ( gcmd_line.display_info == 118 ) {
    printf("\nreachability analysys came up with:");

    printf("\n\npossibly positive facts:");
    for ( f = ginitial; f; f = f->next ) {
      printf("\n");
      print_Fact( f->fact );
    }
    for ( i = 0; i < gnum_relevant_facts; i++ ) {
      printf("\n");
      print_Fact( &(grelevant_facts[i]) );
    }

    printf("\n\nthis yields these %d action templates:", gnum_actions);
    for ( i = 0; i < gnum_operators; i++ ) {
      printf("\n\noperator %s:", goperators[i]->name);
      for ( a = gactions; a; a = a->next ) {
	if ( ( a->norm_operator && 
	       a->norm_operator->operator !=  goperators[i] ) ||
	     ( a->pseudo_action &&
	       a->pseudo_action->operator !=  goperators[i] ) ) {
	  continue;
	}
	printf("\ntemplate: ");
	if ( a->axiom ) printf("(axiom) ");
	for ( j = 0; j < goperators[i]->number_of_real_params; j++ ) {
	  printf("%s", gconstants[a->name_inst_table[j]]);
	  if ( j < goperators[i]->num_vars-1 ) {
	    printf(" ");
	  }
	}
      }
    }
    printf("\n\n");
  }

}



/* bit complicated to avoid memory explosion when high arity predicates take
 * num_obs ^ arity space. take space for individual arg types only; 
 * must consider pred args in smallest - to - largest - type order to make
 * mapping injective.
 */
int fact_adress( void )

{

  int r = 0, b = 1, i, j, min, minj;
  Bool done[MAX_ARITY];

  for ( i = 0; i < garity[lp]; i++ ) {
    done[i] = FALSE;
  }

  for ( i = 0; i < garity[lp]; i++ ) {
    min = -1;
    minj = -1;
    for ( j = 0; j < garity[lp]; j++ ) {
      if ( !done[j] ) {
	if ( min == -1 ||
	     gtype_size[gpredicates_args_type[lp][j]] < min ) {
	  min = gtype_size[gpredicates_args_type[lp][j]];
	  minj = j;
	}
      }
    }
    if ( minj == -1 || min == -1 ) {
      printf("\n\nmin or minj not made in fact adress?\n\n");
      exit( 1 );
    }
    /* now minj is remaining arg with lowest type size min
     */
    /* need number **within type** here! */
    r += b * gmember_nr[largs[minj]][gpredicates_args_type[lp][minj]];
    b *= min;
    done[minj] = TRUE;
  }

  return r;

}



int fluent_adress( void )

{

  int r = 0, b = 1, i;

  for ( i = gf_arity[lf] - 1; i > -1; i-- ) {
    r += b * lf_args[i];
    b *= gnum_constants;
  }

  return r;

}



void make_name_inst_table_from_NormOperator( Action *a, NormOperator *o, EasyTemplate *t )

{

  int i, r = 0, m = 0;

  for ( i = 0; i < o->operator->number_of_real_params; i++ ) {
    if ( o->num_removed_vars > r &&
	 o->removed_vars[r] == i ) {
      /* this var has been removed in NormOp;
       * insert type constraint constant
       *
       * at least one there, as empty typed pars ops are removed
       */
      a->name_inst_table[i] = gtype_consts[o->type_removed_vars[r]][0];
      r++;
    } else {
      /* this par corresponds to par m  in NormOp
       */
      a->name_inst_table[i] = t->inst_table[m];
      m++;
    }
  }

}



void make_name_inst_table_from_PseudoAction( Action *a, PseudoAction *pa )

{

  int i;

  for ( i = 0; i < pa->operator->number_of_real_params; i++ ) {
    a->name_inst_table[i] = pa->inst_table[i];
  }

}


















/***********************************************************
 * RELEVANCE ANALYSIS AND FINAL DOMAIN AND PROBLEM CLEANUP *
 ***********************************************************/









/* counts effects for later allocation
 */
int lnum_effects;









void collect_relevant_facts_and_fluents( void )

{

  Action *a;
  NormOperator *no;
  NormEffect *ne;
  int i, j, adr, size;
  PseudoAction *pa;
  PseudoActionEffect *pae;
  FluentValues *fvs;

  /* facts: mark all deleted facts; such facts, that are also pos, are relevant.
   */
  for ( a = gactions; a; a = a->next ) {
    if ( a->norm_operator ) {
      no = a->norm_operator;

      for ( ne = no->effects; ne; ne = ne->next ) {
	for ( i = 0; i < ne->num_dels; i++ ) {
	  lp = ne->dels[i].predicate;
	  for ( j = 0; j < garity[lp]; j++ ) {
	    largs[j] = ( ne->dels[i].args[j] >= 0 ) ?
	      ne->dels[i].args[j] : a->inst_table[DECODE_VAR( ne->dels[i].args[j] )];
	  }
	  adr = fact_adress();

	  lneg[lp][adr] = 1;
	  if ( lpos[lp][adr] &&
	       !luse[lp][adr] ) {
	    luse[lp][adr] = 1;
	    lindex[lp][adr] = gnum_relevant_facts;
	    if ( gnum_relevant_facts == MAX_RELEVANT_FACTS ) {
	      printf("\nincrease MAX_RELEVANT_FACTS! (current value: %d)\n\n",
		     MAX_RELEVANT_FACTS);
	      exit( 1 );
	    }
	    grelevant_facts[gnum_relevant_facts].predicate = lp;
	    for ( j = 0; j < garity[lp]; j++ ) {
	      grelevant_facts[gnum_relevant_facts].args[j] = largs[j];
	    }
	    lindex[lp][adr] = gnum_relevant_facts;
	    gnum_relevant_facts++;
	  }
	}
      }
    } else {
      pa = a->pseudo_action;

      for ( pae = pa->effects; pae; pae = pae->next ) {
	for ( i = 0; i < pae->num_dels; i++ ) {
	  lp = pae->dels[i].predicate;
	  for ( j = 0; j < garity[lp]; j++ ) {
	    largs[j] = pae->dels[i].args[j];
	  }
	  adr = fact_adress();

	  lneg[lp][adr] = 1;
	  if ( lpos[lp][adr] &&
	       !luse[lp][adr] ) {
	    luse[lp][adr] = 1;
	    lindex[lp][adr] = gnum_relevant_facts;
	    if ( gnum_relevant_facts == MAX_RELEVANT_FACTS ) {
	      printf("\nincrease MAX_RELEVANT_FACTS! (current value: %d)\n\n",
		     MAX_RELEVANT_FACTS);
	      exit( 1 );
	    }
	    grelevant_facts[gnum_relevant_facts].predicate = lp;
	    for ( j = 0; j < garity[lp]; j++ ) {
	      grelevant_facts[gnum_relevant_facts].args[j] = largs[j];
	    }
	    lindex[lp][adr] = gnum_relevant_facts;
	    gnum_relevant_facts++;
	  }
	}
      }
    }
  }
  /* fluents: collect all that are defined in initial state, plus
   * all that are assigned to by an effect of an action
   * (i.e. preconds poss. pos. due to reachability)
   *
   * first initialise fast access structures
   */
  for ( i = 0; i < gnum_functions; i++ ) {
    size =  1;
    for ( j = 0; j < gf_arity[i]; j++ ) {
      size *= gnum_constants;
    }
    lf_def[i] = ( int_pointer ) calloc( size, sizeof( int ) );
    lf_index[i] = ( int_pointer ) calloc( size, sizeof( int ) );
    for ( j = 0; j < size; j++ ) {
      lf_def[i][j] = 0;
      lf_index[i][j] = -1;
    }
  }
  /* from initial state, only those that are not static.
   */
  for ( fvs = gf_initial; fvs; fvs = fvs->next ) {
    lf = fvs->fluent.function;
    for ( j = 0; j < gf_arity[lf]; j++ ) {
      lf_args[j] = fvs->fluent.args[j];
    }
    adr = fluent_adress();
    if ( !lf_def[lf][adr] ) {
      lf_def[lf][adr] = 1;
      if ( gnum_relevant_fluents == MAX_RELEVANT_FLUENTS ) {
	printf("\ntoo many relevant fluents! increase MAX_RELEVANT_FLUENTS (currently %d)\n\n",
	       MAX_RELEVANT_FLUENTS);
	exit( 1 );
      }
      grelevant_fluents[gnum_relevant_fluents].function = lf;
      grelevant_fluents_name[gnum_relevant_fluents] = 
	( char * ) calloc( MAX_LENGTH, sizeof( char ) );
      strcpy( grelevant_fluents_name[gnum_relevant_fluents], gfunctions[lf] );
      for ( j = 0; j < gf_arity[lf]; j++ ) {
	grelevant_fluents[gnum_relevant_fluents].args[j] = lf_args[j];
	strcat( grelevant_fluents_name[gnum_relevant_fluents], "_" );
	strcat( grelevant_fluents_name[gnum_relevant_fluents], gconstants[lf_args[j]] );
      }
      lf_index[lf][adr] = gnum_relevant_fluents;
      gnum_relevant_fluents++;
    } else {
      printf("\n\nfluent ");
      print_Fluent( &(fvs->fluent) );
      printf(" defined twice in initial state! check input files\n\n");
      exit( 1 );
    }
  }
  /* from actions, all assigns (are non-static anyway)
   */
  for ( a = gactions; a; a = a->next ) {
    if ( a->norm_operator ) {
      no = a->norm_operator;
      for ( ne = no->effects; ne; ne = ne->next ) {
	for ( i = 0; i < ne->num_numeric_effects; i++ ) {
	  if ( ne->numeric_effects_neft[i] != ASSIGN ) continue;
	  lf = ne->numeric_effects_fluent[i].function;
	  for ( j = 0; j < gf_arity[lf]; j++ ) {
	    lf_args[j] = ( ne->numeric_effects_fluent[i].args[j] >= 0 ) ?
	      ne->numeric_effects_fluent[i].args[j] : 
	      a->inst_table[DECODE_VAR( ne->numeric_effects_fluent[i].args[j] )];
	  }
	  adr = fluent_adress();
	  if ( !lf_def[lf][adr] ) {
	    lf_def[lf][adr] = 1;
	    if ( gnum_relevant_fluents == MAX_RELEVANT_FLUENTS ) {
	      printf("\ntoo many relevant fluents! increase MAX_RELEVANT_FLUENTS (currently %d)\n\n",
		     MAX_RELEVANT_FLUENTS);
	      exit( 1 );
	    }
	    grelevant_fluents[gnum_relevant_fluents].function = lf;
	    grelevant_fluents_name[gnum_relevant_fluents] = 
	      ( char * ) calloc( MAX_LENGTH, sizeof( char ) );
	    strcpy( grelevant_fluents_name[gnum_relevant_fluents], gfunctions[lf] );
	    for ( j = 0; j < gf_arity[lf]; j++ ) {
	      grelevant_fluents[gnum_relevant_fluents].args[j] = lf_args[j];
	      strcat( grelevant_fluents_name[gnum_relevant_fluents], "_" );
	      strcat( grelevant_fluents_name[gnum_relevant_fluents], gconstants[lf_args[j]] );
	    }
	    lf_index[lf][adr] = gnum_relevant_fluents;
	    gnum_relevant_fluents++;
	  }
	}
      }
    } else {
      pa = a->pseudo_action;
      for ( pae = pa->effects; pae; pae = pae->next ) {
	for ( i = 0; i < pae->num_numeric_effects; i++ ) {
	  if ( pae->numeric_effects_neft[i] != ASSIGN ) continue;
	  lf = pae->numeric_effects_fluent[i].function;
	  for ( j = 0; j < gf_arity[lf]; j++ ) {
	    lf_args[j] = ( pae->numeric_effects_fluent[i].args[j] >= 0 ) ?
	      pae->numeric_effects_fluent[i].args[j] : 
	      a->inst_table[DECODE_VAR( pae->numeric_effects_fluent[i].args[j] )];
	  }
	  adr = fluent_adress();
	  if ( !lf_def[lf][adr] ) {
	    lf_def[lf][adr] = 1;
	    if ( gnum_relevant_fluents == MAX_RELEVANT_FLUENTS ) {
	      printf("\ntoo many relevant fluents! increase MAX_RELEVANT_FLUENTS (currently %d)\n\n",
		     MAX_RELEVANT_FLUENTS);
	      exit( 1 );
	    }
	    grelevant_fluents[gnum_relevant_fluents].function = lf;
	    grelevant_fluents_name[gnum_relevant_fluents] = 
	      ( char * ) calloc( MAX_LENGTH, sizeof( char ) );
	    strcpy( grelevant_fluents_name[gnum_relevant_fluents], gfunctions[lf] );
	    for ( j = 0; j < gf_arity[lf]; j++ ) {
	      grelevant_fluents[gnum_relevant_fluents].args[j] = lf_args[j];
	      strcat( grelevant_fluents_name[gnum_relevant_fluents], "_" );
	      strcat( grelevant_fluents_name[gnum_relevant_fluents], gconstants[lf_args[j]] );
	    }
	    lf_index[lf][adr] = gnum_relevant_fluents;
	    gnum_relevant_fluents++;
	  }
	}
      }
    }
  }

  if ( gcmd_line.display_info == 119 ) {
    printf("\n\nfacts selected as relevant:");
    for ( i = 0; i < gnum_relevant_facts; i++ ) {
      printf("\n%d: ", i);
      print_Fact( &(grelevant_facts[i]) );
    }
    printf("\n\nfluents selected as relevant:");
    for ( i = 0; i < gnum_relevant_fluents; i++ ) {
      printf("\n%d: ", i);
      print_Fluent( &(grelevant_fluents[i]) );
    }    
    printf("\n\n");
  }

  lnum_effects = 0;

  create_final_goal_state();
  create_final_initial_state();
  create_final_actions();

  if ( gmetric != NULL ) {
    if ( !set_relevants_in_exp( &gmetric ) ) {
      if ( gcmd_line.display_info ) {
	printf("\nwarning: undefined fluent used in optimization expression. defaulting to plan length");
      }
      free_ExpNode( gmetric );
      gmetric = NULL;      
    }
  }

  if ( gcmd_line.display_info == 120 ) {
    printf("\n\nfinal domain representation is:\n\n");  

    for ( i = 0; i < gnum_operators; i++ ) {
      printf("\n\n------------------operator %s-----------\n\n", goperators[i]->name);
      for ( a = gactions; a; a = a->next ) {
	if ( ( !a->norm_operator &&
	       !a->pseudo_action ) ||
	     ( a->norm_operator && 
	       a->norm_operator->operator != goperators[i] ) ||
	     ( a->pseudo_action &&
	       a->pseudo_action->operator != goperators[i] ) ) {
	  continue;
	}
	print_Action( a );
      }
    }
    printf("\n\n--------------------GOAL REACHED ops-----------\n\n");
    for ( a = gactions; a; a = a->next ) {
      if ( !a->norm_operator &&
	   !a->pseudo_action ) {
	print_Action( a );
      }
    }

    printf("\n\nfinal initial state is:\n\n");
    print_State( ginitial_state );

    printf("\n\nfinal goal is:\n\n");
    for ( i = 0; i < gnum_logic_goal; i++ ) {
      print_ft_name( glogic_goal[i] );
      printf("\n");
    }
    for ( i = 0; i < gnum_numeric_goal; i++ ) {
      switch ( gnumeric_goal_comp[i] ) {
      case LE:
	printf("(< ");
	break;
      case LEQ:
	printf("(<= ");
	break;
      case EQ:
	printf("(= ");
	break;
      case GEQ:
	printf("(>= ");
	break;
      case GE:
	printf("(> ");
	break;
      default:
	printf("\nwrong comparator in gnumeric_goal %d\n\n", gnumeric_goal_comp[i]);
	exit( 1 );
      }
      print_ExpNode( gnumeric_goal_lh[i] );
      print_ExpNode( gnumeric_goal_rh[i] );
      printf(")\n");
    }

    if ( gmetric ) {
      printf("\n\nmetric is (minimize):\n");
      print_ExpNode( gmetric );
    } else {
      printf("\n\nmetric: none, i.e. plan length\n");
    }
  }

}



void create_final_goal_state( void )

{

  WffNode *w, *ww;
  int m, mn, i, adr;
  Action *tmp;

  if ( !set_relevants_in_wff( &ggoal ) ) {
    printf("\n\nff: goal accesses a fluent that will never have a defined value. Problem unsolvable.\n\n");
    exit( 1 );
  }
  cleanup_wff( &ggoal );

  if ( ggoal->connective == TRU ) {
    printf("\nff: goal can be simplified to TRUE. The empty plan solves it\n\n");
    gnum_plan_ops = 0;
    exit( 1 );
  }
  if ( ggoal->connective == FAL ) {
    printf("\nff: goal can be simplified to FALSE. No plan will solve it\n\n");
    exit( 1 );
  }

  switch ( ggoal->connective ) {
  case OR:
    if ( gnum_relevant_facts == MAX_RELEVANT_FACTS ) {
      printf("\nincrease MAX_RELEVANT_FACTS! (current value: %d)\n\n",
	     MAX_RELEVANT_FACTS);
      exit( 1 );
    }
    grelevant_facts[gnum_relevant_facts].predicate = -3;
    gnum_relevant_facts++;
    for ( w = ggoal->sons; w; w = w->next ) {
      tmp = new_Action();
      if ( w->connective == AND ) {
	m = 0; mn = 0;
	for ( ww = w->sons; ww; ww = ww->next ) {
	  if ( ww->connective == ATOM ) m++;
	  if ( ww->connective == COMP ) mn++;
	}
	tmp->preconds = ( int * ) calloc( m, sizeof( int ) );
	tmp->numeric_preconds_comp = ( Comparator * ) calloc( mn, sizeof( Comparator ) );
	tmp->numeric_preconds_lh = ( ExpNode_pointer * ) calloc( mn, sizeof( ExpNode_pointer ) );
	tmp->numeric_preconds_rh = ( ExpNode_pointer * ) calloc( mn, sizeof( ExpNode_pointer ) );
	tmp->num_preconds = m;
	tmp->num_numeric_preconds = mn;
	m = 0; mn = 0;
	for ( ww = w->sons; ww; ww = ww->next ) {
	  if ( ww->connective == ATOM ) {
	    lp = ww->fact->predicate;
	    for ( i = 0; i < garity[lp]; i++ ) {
	      largs[i] = ww->fact->args[i];
	    }
	    adr = fact_adress();
	    tmp->preconds[m] = lindex[lp][adr];
	    m++;
	  }
	  if ( ww->connective == COMP ) {
	    tmp->numeric_preconds_comp[mn] = ww->comp;
	    tmp->numeric_preconds_lh[mn] = copy_Exp( ww->lh );
	    tmp->numeric_preconds_rh[mn] = copy_Exp( ww->rh );	    
	    mn++;
	  }
	}
      } else {
	if ( w->connective == ATOM ) {
	  tmp->preconds = ( int * ) calloc( 1, sizeof( int ) );
	  tmp->num_preconds = 1;
	  lp = w->fact->predicate;
	  for ( i = 0; i < garity[lp]; i++ ) {
	    largs[i] = w->fact->args[i];
	  }
	  adr = fact_adress();
	  tmp->preconds[0] = lindex[lp][adr];
	}
	if ( w->connective == COMP ) {
	  tmp->numeric_preconds_comp = ( Comparator * ) calloc( 1, sizeof( Comparator ) );
	  tmp->numeric_preconds_lh = ( ExpNode_pointer * ) calloc( 1, sizeof( ExpNode_pointer ) );
	  tmp->numeric_preconds_rh = ( ExpNode_pointer * ) calloc( 1, sizeof( ExpNode_pointer ) );
	  tmp->numeric_preconds_comp[0] = w->comp;
	  tmp->numeric_preconds_lh[0] = copy_Exp( w->lh );
	  tmp->numeric_preconds_rh[0] = copy_Exp( w->rh );
	  tmp->num_numeric_preconds = 1;
	}
      }
      tmp->effects = ( ActionEffect * ) calloc( 1, sizeof( ActionEffect ) );
      tmp->num_effects = 1;
      tmp->effects[0].conditions = NULL;
      tmp->effects[0].num_conditions = 0;
      tmp->effects[0].dels = NULL;
      tmp->effects[0].num_dels = 0;
      tmp->effects[0].adds = ( int * ) calloc( 1, sizeof( int ) );
      tmp->effects[0].adds[0] = gnum_relevant_facts - 1;
      tmp->effects[0].num_adds = 1;
      tmp->effects[0].numeric_conditions_comp = NULL;
      tmp->effects[0].numeric_conditions_lh = NULL;
      tmp->effects[0].numeric_conditions_rh = NULL;
      tmp->effects[0].num_numeric_conditions = 0;
      tmp->effects[0].numeric_effects_neft = NULL;
      tmp->effects[0].numeric_effects_fl = NULL;
      tmp->effects[0].numeric_effects_rh = NULL;
      tmp->effects[0].num_numeric_effects = 0;

      tmp->next = gactions;
      gactions = tmp;
      gnum_actions++;
      lnum_effects++;
    }
    glogic_goal = ( int * ) calloc( 1, sizeof( int ) );
    glogic_goal[0] = gnum_relevant_facts - 1;
    gnum_logic_goal = 1;
    break;
  case AND:
    m = 0; mn = 0;
    for ( w = ggoal->sons; w; w = w->next ) {
      if ( w->connective == ATOM ) m++;
      if ( w->connective == COMP ) mn++;
    }
    glogic_goal = ( int * ) calloc( m, sizeof( int ) );
    gnumeric_goal_comp = ( Comparator * ) calloc( mn, sizeof( Comparator ) );
    gnumeric_goal_lh = ( ExpNode_pointer * ) calloc( mn, sizeof( ExpNode_pointer ) );
    gnumeric_goal_rh = ( ExpNode_pointer * ) calloc( mn, sizeof( ExpNode_pointer ) );
    gnum_logic_goal = m;
    gnum_numeric_goal = mn;
    m = 0; mn = 0;
    for ( w = ggoal->sons; w; w = w->next ) {
      if ( w->connective == ATOM ) {
	lp = w->fact->predicate;
	for ( i = 0; i < garity[lp]; i++ ) {
	  largs[i] = w->fact->args[i];
	}
	adr = fact_adress();
	glogic_goal[m] = lindex[lp][adr];
	m++;
      }
      if ( w->connective == COMP ) {
	gnumeric_goal_comp[mn] = w->comp;
	gnumeric_goal_lh[mn] = copy_Exp( w->lh ); 
	gnumeric_goal_rh[mn] = copy_Exp( w->rh );
	mn++;
      }
    }
    break;
  case ATOM:
    glogic_goal = ( int * ) calloc( 1, sizeof( int ) );
    gnum_logic_goal = 1;
    lp = ggoal->fact->predicate;
    for ( i = 0; i < garity[lp]; i++ ) {
      largs[i] = ggoal->fact->args[i];
    }
    adr = fact_adress();
    glogic_goal[0] = lindex[lp][adr];
    break;
  case COMP:
    gnumeric_goal_comp = ( Comparator * ) calloc( 1, sizeof( Comparator ) );
    gnumeric_goal_lh = ( ExpNode_pointer * ) calloc( 1, sizeof( ExpNode_pointer ) );
    gnumeric_goal_rh = ( ExpNode_pointer * ) calloc( 1, sizeof( ExpNode_pointer ) );
    gnum_numeric_goal = 1;
    gnumeric_goal_comp[0] = ggoal->comp;
    gnumeric_goal_lh[0] = copy_Exp( ggoal->lh ); 
    gnumeric_goal_rh[0] = copy_Exp( ggoal->rh );
    break;
  default:
    printf("\n\nwon't get here: non COMP,ATOM,AND,OR in fully simplified goal\n\n");
    exit( 1 );
  }

}



Bool set_relevants_in_wff( WffNode **w )

{

  WffNode *i;
  int j, adr;

  switch ( (*w)->connective ) {
  case AND:
  case OR:
    for ( i = (*w)->sons; i; i = i->next ) {
      if ( !set_relevants_in_wff( &i ) ) {
	return FALSE;
      }
    }
    break;
  case ATOM:
    /* no equalities, as fully instantiated
     */
    lp = (*w)->fact->predicate;
    for ( j = 0; j < garity[lp]; j++ ) {
      largs[j] = (*w)->fact->args[j];
    }
    adr = fact_adress();

    if ( !lneg[lp][adr] ) {
      (*w)->connective = TRU;
      free( (*w)->fact );
      (*w)->fact = NULL;
      break;
    }
    if ( !lpos[lp][adr] ) {
      (*w)->connective = FAL;
      free( (*w)->fact );
      (*w)->fact = NULL;
      break;
    }
    break;
  case COMP:
    if ( !set_relevants_in_exp( &((*w)->lh) ) ||
	 !set_relevants_in_exp( &((*w)->rh) ) ) {
      return FALSE;
    }
    break;
  default:
    printf("\n\nwon't get here: non ATOM,OR,AND in goal set relevants\n\n");
    exit( 1 );
  }

  return TRUE;

}



Bool set_relevants_in_exp( ExpNode **n )

{

  int j, adr;

  /* can probably (for sure) forget about the simplification
   * stuff here because it's been done before.
   *
   * igual....
   */
  switch ( (*n)->connective ) {
  case AD:
    if ( !set_relevants_in_exp( &((*n)->leftson) ) ) return FALSE;
    if ( !set_relevants_in_exp( &((*n)->rightson) ) ) return FALSE;
    if ( (*n)->leftson->connective != NUMBER ||
	 (*n)->rightson->connective != NUMBER ) {
      break;
    }
    (*n)->connective = NUMBER;
    (*n)->value = (*n)->leftson->value + (*n)->rightson->value;
    free_ExpNode( (*n)->leftson );
    (*n)->leftson = NULL;
    free_ExpNode( (*n)->rightson );
    (*n)->rightson = NULL;
    break;
  case SU:
    if ( !set_relevants_in_exp( &((*n)->leftson) ) ) return FALSE;
    if ( !set_relevants_in_exp( &((*n)->rightson) ) ) return FALSE;
    if ( (*n)->leftson->connective != NUMBER ||
	 (*n)->rightson->connective != NUMBER ) {
      break;
    }
    (*n)->connective = NUMBER;
    (*n)->value = (*n)->leftson->value - (*n)->rightson->value;
    free_ExpNode( (*n)->leftson );
    (*n)->leftson = NULL;
    free_ExpNode( (*n)->rightson );
    (*n)->rightson = NULL;
    break;
  case MU:
    if ( !set_relevants_in_exp( &((*n)->leftson) ) ) return FALSE;
    if ( !set_relevants_in_exp( &((*n)->rightson) ) ) return FALSE;
    if ( (*n)->leftson->connective != NUMBER ||
	 (*n)->rightson->connective != NUMBER ) {
      break;
    }
    (*n)->connective = NUMBER;
    (*n)->value = (*n)->leftson->value * (*n)->rightson->value;
    free_ExpNode( (*n)->leftson );
    (*n)->leftson = NULL;
    free_ExpNode( (*n)->rightson );
    (*n)->rightson = NULL;
    break;
  case DI:
    if ( !set_relevants_in_exp( &((*n)->leftson) ) ) return FALSE;
    if ( !set_relevants_in_exp( &((*n)->rightson) ) ) return FALSE;
    if ( (*n)->leftson->connective != NUMBER ||
	 (*n)->rightson->connective != NUMBER ) {
      break;
    }
    if ( (*n)->rightson->value == 0 ) {
      /* kind of unclean: simply leave that in here;
       * we will later determine the right thing 
       * to do with it.
       */
      break;
    }
    (*n)->connective = NUMBER;
    (*n)->value = (*n)->leftson->value / (*n)->rightson->value;
    free_ExpNode( (*n)->leftson );
    (*n)->leftson = NULL;
    free_ExpNode( (*n)->rightson );
    (*n)->rightson = NULL;
    break;
  case MINUS:
    if ( !set_relevants_in_exp( &((*n)->son) ) ) return FALSE;
    if ( (*n)->son->connective != NUMBER ) break;
    (*n)->connective = NUMBER;
    (*n)->value = ((float) (-1)) * (*n)->son->value;
    free_ExpNode( (*n)->son );
    (*n)->son = NULL;
    break;    
  case NUMBER:
    break;
  case FHEAD:
    lf = (*n)->fluent->function;
    for ( j = 0; j < gf_arity[lf]; j++ ) {
      lf_args[j] = (*n)->fluent->args[j];
    }
    adr = fluent_adress();
    (*n)->fl = lf_index[lf][adr];
    free( (*n)->fluent );
    (*n)->fluent = NULL;
    if ( lf_index[lf][adr] == -1 ) {
      if ( lf == 0 ) {
	/* ATTENTION!! FUNCTION 0 IS TOTAL-TIME WHICH IS *ONLY* USED
	 * IN OPTIMIZATION EXPRESSION. GETS A SPECIAL TREATMENT
	 * IN THE RESPECTIVE FUNCTION IN SEARCH.C!!!!
	 *
	 * we remember it as fluent -2!!
	 */
	(*n)->fl = -2;
      } else {
	return FALSE;
      }
    }
    break;
  default:
    printf("\n\nset relevants in expnode: wrong specifier %d",
	   (*n)->connective);
    exit( 1 );
  }

  return TRUE;

}



void create_final_initial_state( void )

{

  Facts *f;
  int i, adr, fl;
  FluentValues *fvs;

  i = 0;
/*   for ( f = ginitial; f; f = f->next ) i++; */
  /* we need space for transformation fluents to come!
   *
   * ALSO, we may need space for derived facts!!!
   */
  make_state( &ginitial_state, gnum_relevant_facts + 1, MAX_RELEVANT_FLUENTS );

  for ( f = ginitial; f; f = f->next ) {
    lp = f->fact->predicate;
    for ( i = 0; i < garity[lp]; i++ ) {
      largs[i] = f->fact->args[i];
    }
    adr = fact_adress();
    if ( !lneg[lp][adr] ) {/* non deleted ini */
      continue;
    }
    ginitial_state.F[ginitial_state.num_F++] = lindex[lp][adr];
  }

  for ( fvs = gf_initial; fvs; fvs = fvs->next ) {
    lf = fvs->fluent.function;
    for ( i = 0; i < gf_arity[lf]; i++ ) {
      lf_args[i] = fvs->fluent.args[i];
    }
    adr = fluent_adress();
    fl = lf_index[lf][adr];
    ginitial_state.f_D[fl] = TRUE;
    ginitial_state.f_V[fl] = fvs->value;
  }

}



void create_final_actions( void )

{

  Action *a, *p, *t;
  NormOperator *no;
  NormEffect *ne;
  int i, j, adr;
  PseudoAction *pa;
  PseudoActionEffect *pae;
  ActionEffect *aa;
  Bool false_cond;

  a = gactions; p = NULL;
  while ( a ) {
    if ( a->norm_operator ) {
      /* action comes from an easy template NormOp
       */
      no = a->norm_operator;

      if ( no->num_preconds > 0 ) {
	a->preconds = ( int * ) calloc( no->num_preconds, sizeof( int ) );
      }
      a->num_preconds = 0;
      for ( i = 0; i < no->num_preconds; i++ ) {
	lp = no->preconds[i].predicate;
	for ( j = 0; j < garity[lp]; j++ ) {
	  largs[j] = ( no->preconds[i].args[j] >= 0 ) ?
	    no->preconds[i].args[j] : a->inst_table[DECODE_VAR( no->preconds[i].args[j] )];
	}
	adr = fact_adress();	
	/* preconds are lpos in all cases due to reachability analysis
	 */
	if ( !lneg[lp][adr] ) {
	  continue;
	}
	a->preconds[a->num_preconds++] = lindex[lp][adr];
      }

      /**************************NUMERIC PRECOND*************************/
      if ( no->num_numeric_preconds > 0 ) {
	a->numeric_preconds_comp = ( Comparator * ) 
	  calloc( no->num_numeric_preconds, sizeof( Comparator ) );
	a->numeric_preconds_lh = ( ExpNode_pointer * )
	  calloc( no->num_numeric_preconds, sizeof( ExpNode_pointer ) );
	a->numeric_preconds_rh = ( ExpNode_pointer * )
	  calloc( no->num_numeric_preconds, sizeof( ExpNode_pointer ) );
	a->num_numeric_preconds = 0;
      }
      for ( i = 0; i < no->num_numeric_preconds; i++ ) {
	a->numeric_preconds_comp[a->num_numeric_preconds] = no->numeric_preconds_comp[i];
	a->numeric_preconds_lh[a->num_numeric_preconds] = copy_Exp( no->numeric_preconds_lh[i] );
	instantiate_exp_by_action( &(a->numeric_preconds_lh[a->num_numeric_preconds]), a );
	if ( !set_relevants_in_exp( &(a->numeric_preconds_lh[a->num_numeric_preconds]) ) ) break;
	a->numeric_preconds_rh[a->num_numeric_preconds] = copy_Exp( no->numeric_preconds_rh[i] );
	instantiate_exp_by_action( &(a->numeric_preconds_rh[a->num_numeric_preconds]), a );
	if ( !set_relevants_in_exp( &(a->numeric_preconds_rh[a->num_numeric_preconds]) ) ) break;
	if ( a->numeric_preconds_lh[a->num_numeric_preconds]->connective == NUMBER &&
	     a->numeric_preconds_rh[a->num_numeric_preconds]->connective == NUMBER ) {
	  /* trivial numeric precond
	   */
	  if ( number_comparison_holds( a->numeric_preconds_comp[a->num_numeric_preconds],
					a->numeric_preconds_lh[a->num_numeric_preconds]->value,
					a->numeric_preconds_rh[a->num_numeric_preconds]->value ) ) {
	    /* true precond -> throw precond away. by not incrementing number of such.
	     */
	    free_ExpNode( a->numeric_preconds_lh[a->num_numeric_preconds] );
	    free_ExpNode( a->numeric_preconds_rh[a->num_numeric_preconds] );
	    continue;
	  } else {
	    /* false precond -> throw action away.
	     */
	    break;
	  }
	}
	a->num_numeric_preconds++;
      }
      if ( i < no->num_numeric_preconds ) {
	/* a precond accesses an undefined fluent, or is false -> remove action!
	 */
	gnum_actions--;
	if ( p ) {
	  p->next = a->next;
	  t = a;
	  a = a->next;
	  t->next = gtrash_actions;
	  gtrash_actions = t;
	} else {
	  gactions = a->next;
	  t = a;
	  a = a->next;
	  t->next = gtrash_actions;
	  gtrash_actions = t;
	}
	continue;
      }
      /**************************NUMERIC PRECOND-END*************************/

      /* and now for the effects
       */
      if ( a->num_effects > 0 ) {
	a->effects = ( ActionEffect * ) calloc( a->num_effects, sizeof( ActionEffect ) );
	for ( i = 0; i < a->num_effects; i++ ) {
	  a->effects[i].illegal = FALSE;
	  a->effects[i].removed = FALSE;
	}
      }
      a->num_effects = 0;
      for ( ne = no->effects; ne; ne = ne->next ) {
	aa = &(a->effects[a->num_effects]);

	if ( ne->num_conditions > 0 ) {
	  aa->conditions = ( int * ) calloc( ne->num_conditions, sizeof( int ) );
	}
	aa->num_conditions = 0;
	for ( i = 0; i < ne->num_conditions; i++ ) {
	  lp = ne->conditions[i].predicate;
	  for ( j = 0; j < garity[lp]; j++ ) {
	    largs[j] = ( ne->conditions[i].args[j] >= 0 ) ?
	      ne->conditions[i].args[j] : a->inst_table[DECODE_VAR( ne->conditions[i].args[j] )];
	  }
	  adr = fact_adress();
	  if ( !lpos[lp][adr] ) {/* condition not reachable: skip effect */
	    break;
	  }
	  if ( !lneg[lp][adr] ) {/* condition always true: skip it */
	    continue;
	  }
	  aa->conditions[aa->num_conditions++] = lindex[lp][adr];
	}
	if ( i < ne->num_conditions ) {/* found unreachable condition: free condition space */
	  free( aa->conditions );
	  continue;
	}

	/**************************NUMERIC COND*************************/
	if ( ne->num_numeric_conditions > 0 ) {
	  aa->numeric_conditions_comp = ( Comparator * ) 
	    calloc( ne->num_numeric_conditions, sizeof( Comparator ) );
	  aa->numeric_conditions_lh = ( ExpNode_pointer * )
	    calloc( ne->num_numeric_conditions, sizeof( ExpNode_pointer ) );
	  aa->numeric_conditions_rh = ( ExpNode_pointer * )
	    calloc( ne->num_numeric_conditions, sizeof( ExpNode_pointer ) );
	  for ( i = 0; i < ne->num_numeric_conditions; i++ ) {
	    aa->numeric_conditions_lh[i] = NULL;
	    aa->numeric_conditions_rh[i] = NULL;
	  }
	  aa->num_numeric_conditions = 0;
	}
	false_cond = FALSE;
	for ( i = 0; i < ne->num_numeric_conditions; i++ ) {
	  aa->numeric_conditions_comp[aa->num_numeric_conditions] = ne->numeric_conditions_comp[i];
	  aa->numeric_conditions_lh[aa->num_numeric_conditions] = copy_Exp( ne->numeric_conditions_lh[i] );
	  instantiate_exp_by_action( &(aa->numeric_conditions_lh[aa->num_numeric_conditions]), a );
	  if ( !set_relevants_in_exp( &(aa->numeric_conditions_lh[aa->num_numeric_conditions]) ) ) break;
	  aa->numeric_conditions_rh[aa->num_numeric_conditions] = copy_Exp( ne->numeric_conditions_rh[i] );
	  instantiate_exp_by_action( &(aa->numeric_conditions_rh[aa->num_numeric_conditions]), a );
	  if ( !set_relevants_in_exp( &(aa->numeric_conditions_rh[aa->num_numeric_conditions]) ) ) break;
	  if ( aa->numeric_conditions_lh[aa->num_numeric_conditions]->connective == NUMBER &&
	       aa->numeric_conditions_rh[aa->num_numeric_conditions]->connective == NUMBER ) {
	    /* trivial numeric condition
	     */
	    if ( number_comparison_holds( aa->numeric_conditions_comp[aa->num_numeric_conditions],
					  aa->numeric_conditions_lh[aa->num_numeric_conditions]->value,
					  aa->numeric_conditions_rh[aa->num_numeric_conditions]->value ) ) {
	      /* true cond -> throw cond away. by not incrementing number of such.
	       */
	      free_ExpNode( aa->numeric_conditions_lh[aa->num_numeric_conditions] );
	      free_ExpNode( aa->numeric_conditions_rh[aa->num_numeric_conditions] );
	      aa->numeric_conditions_lh[aa->num_numeric_conditions] = NULL;
	      aa->numeric_conditions_rh[aa->num_numeric_conditions] = NULL;
	      continue;
	    } else {
	      /* false cond -> throw effect away.
	       */
	      false_cond = TRUE;
	      break;
	    }
	  }
	  aa->num_numeric_conditions++;
	}
	if ( i < ne->num_numeric_conditions ) {
	  if ( false_cond ) {
	    /* false numeric cond: free what's been done so far, and skip effect
	     */
	    for ( i = 0; i <= aa->num_numeric_conditions; i++ ) {
	      free_ExpNode( aa->numeric_conditions_lh[i] );
	      free_ExpNode( aa->numeric_conditions_rh[i] );
	    }
	    free( aa->numeric_conditions_comp );
	    free( aa->numeric_conditions_lh );
	    free( aa->numeric_conditions_rh );
	    continue;/* next effect, without incrementing action counter */
	  } else {
	    /* numeric effect uses undefined fluent in condition -->
	     * THROW WHOLE ACTION AWAY! done by breaking out of the 
	     * effects loop, which will be catched below overall
	     * effect handling.
	     */
	    break;
	  }
	}
	/**************************NUMERIC COND - END*************************/

	/* now create the add and del effects.
	 */
	if ( ne->num_adds > 0 ) {
	  aa->adds = ( int * ) calloc( ne->num_adds, sizeof( int ) );
	}
	aa->num_adds = 0;
	for ( i = 0; i < ne->num_adds; i++ ) {
	  lp = ne->adds[i].predicate;
	  for ( j = 0; j < garity[lp]; j++ ) {
	    largs[j] = ( ne->adds[i].args[j] >= 0 ) ?
	      ne->adds[i].args[j] : a->inst_table[DECODE_VAR( ne->adds[i].args[j] )];
	  }
	  adr = fact_adress();
	  if ( !lneg[lp][adr] ) {/* effect always true: skip it */
	    continue;
	  }
	  aa->adds[aa->num_adds++] = lindex[lp][adr];
	}

	if ( ne->num_dels > 0 ) {
	  aa->dels = ( int * ) calloc( ne->num_dels, sizeof( int ) );
	}
	aa->num_dels = 0;
	for ( i = 0; i < ne->num_dels; i++ ) {
	  lp = ne->dels[i].predicate;
	  for ( j = 0; j < garity[lp]; j++ ) {
	    largs[j] = ( ne->dels[i].args[j] >= 0 ) ?
	      ne->dels[i].args[j] : a->inst_table[DECODE_VAR( ne->dels[i].args[j] )];
	  }
	  adr = fact_adress();
	  if ( !lpos[lp][adr] ) {/* effect always false: skip it */
	    continue;
	  }
	  /* NO CHECK FOR ADD \CAP DEL!!!!! -> ALLOWED BY SEMANTICS!!!
	   */
	  aa->dels[aa->num_dels++] = lindex[lp][adr];
	}
	if ( i < ne->num_dels ) break;

	/**************************NUMERIC EFFECTS*************************/
	if ( ne->num_numeric_effects > 0 ) {
	  aa->numeric_effects_neft = ( NumericEffectType * ) 
	    calloc( ne->num_numeric_effects, sizeof( NumericEffectType ) );
	  aa->numeric_effects_fl = ( int * )
	    calloc( ne->num_numeric_effects, sizeof( int ) );
	  aa->numeric_effects_rh = ( ExpNode_pointer * )
	    calloc( ne->num_numeric_effects, sizeof( ExpNode_pointer ) );
	  aa->num_numeric_effects = 0;
	}
	for ( i = 0; i < ne->num_numeric_effects; i++ ) {
	  aa->numeric_effects_neft[aa->num_numeric_effects] = ne->numeric_effects_neft[i];
	  lf = ne->numeric_effects_fluent[i].function;
	  for ( j = 0; j < gf_arity[lf]; j++ ) {
	    lf_args[j] = ( ne->numeric_effects_fluent[i].args[j] >= 0 ) ?
	      ne->numeric_effects_fluent[i].args[j] : 
	      a->inst_table[DECODE_VAR( ne->numeric_effects_fluent[i].args[j] )];
	  }
	  adr = fluent_adress();
	  /* if it's -1, simply let it in --- if that effect appears, then 
	   * action is illegal, otherwise not.
	   */
	  aa->numeric_effects_fl[i] = lf_index[lf][adr];
	  if ( lf_index[lf][adr] == -1 ) aa->illegal = TRUE;
	  aa->numeric_effects_rh[aa->num_numeric_effects] = copy_Exp( ne->numeric_effects_rh[i] );
	  instantiate_exp_by_action( &(aa->numeric_effects_rh[aa->num_numeric_effects]), a );
	  if ( !set_relevants_in_exp( &(aa->numeric_effects_rh[aa->num_numeric_effects]) ) ) {
	    aa->illegal = TRUE;
	  }
	  if ( aa->illegal &&
	       aa->num_conditions == 0 &&
	       aa->num_numeric_conditions == 0 ) {
	    break;
	  }
	  /* that's it ???????????????? - !!
	   */
	  aa->num_numeric_effects++;
	}
	if ( i < ne->num_numeric_effects ) {
	  /* an unconditional illegal effekt
	   */
	  break;
	}
	/**************************NUMERIC EFFECTS - END*************************/

	/* this effect is OK. go to next one in NormOp.
	 */
	a->num_effects++;
	lnum_effects++;
      }
      if ( ne ) {
	/* we get here if one effect was faulty
	 */
	gnum_actions--;
	if ( p ) {
	  p->next = a->next;
	  t = a;
	  a = a->next;
	  t->next = gtrash_actions;
	  gtrash_actions = t;
	} else {
	  gactions = a->next;
	  t = a;
	  a = a->next;
	  t->next = gtrash_actions;
	  gtrash_actions = t;
	}
      } else {
	p = a;
	a = a->next;
      }
      continue;
    }
    /**********************************second half: hard operators --> pseudo actions******************/
    if ( a->pseudo_action ) {
      /* action is result of a PseudoAction
       */
      pa = a->pseudo_action;
      if ( pa->num_preconds > 0 ) {
	a->preconds = ( int * ) calloc( pa->num_preconds, sizeof( int ) );
      }
      a->num_preconds = 0;
      for ( i = 0; i < pa->num_preconds; i++ ) {
	lp = pa->preconds[i].predicate;
	for ( j = 0; j < garity[lp]; j++ ) {
	  largs[j] = pa->preconds[i].args[j];
	}
	adr = fact_adress();
	/* preconds are lpos in all cases due to reachability analysis
	 */
	if ( !lneg[lp][adr] ) {
	  continue;
	}	
	a->preconds[a->num_preconds++] = lindex[lp][adr];
      }

      /**************************NUMERIC PRECOND*************************/
      if ( pa->num_numeric_preconds > 0 ) {
	a->numeric_preconds_comp = ( Comparator * ) 
	  calloc( pa->num_numeric_preconds, sizeof( Comparator ) );
	a->numeric_preconds_lh = ( ExpNode_pointer * )
	  calloc( pa->num_numeric_preconds, sizeof( ExpNode_pointer ) );
	a->numeric_preconds_rh = ( ExpNode_pointer * )
	  calloc( pa->num_numeric_preconds, sizeof( ExpNode_pointer ) );
	a->num_numeric_preconds = 0;
      }
      for ( i = 0; i < pa->num_numeric_preconds; i++ ) {
	a->numeric_preconds_comp[a->num_numeric_preconds] = pa->numeric_preconds_comp[i];
	a->numeric_preconds_lh[a->num_numeric_preconds] = copy_Exp( pa->numeric_preconds_lh[i] );
	if ( !set_relevants_in_exp( &(a->numeric_preconds_lh[a->num_numeric_preconds]) ) ) break;
	a->numeric_preconds_rh[a->num_numeric_preconds] = copy_Exp( pa->numeric_preconds_rh[i] );
	if ( !set_relevants_in_exp( &(a->numeric_preconds_rh[a->num_numeric_preconds]) ) ) break;
	a->num_numeric_preconds++;
      }
      if ( i < pa->num_numeric_preconds ) {
	/* a precond accesses an undefined fluent -> remove action!
	 */
	gnum_actions--;
	if ( p ) {
	  p->next = a->next;
	  t = a;
	  a = a->next;
	  t->next = gtrash_actions;
	  gtrash_actions = t;
	} else {
	  gactions = a->next;
	  t = a;
	  a = a->next;
	  t->next = gtrash_actions;
	  gtrash_actions = t;
	}
	continue;
      }
      /**************************NUMERIC PRECOND-END*************************/

      /* and now for the effects
       */
      if ( a->num_effects > 0 ) {
	a->effects = ( ActionEffect * ) calloc( a->num_effects, sizeof( ActionEffect ) );
	for ( i = 0; i < a->num_effects; i++ ) {
	  a->effects[i].illegal = FALSE;
	  a->effects[i].removed = FALSE;
	}
      }
      a->num_effects = 0;
      for ( pae = pa->effects; pae; pae = pae->next ) {
	aa = &(a->effects[a->num_effects]);

	if ( pae->num_conditions > 0 ) {
	  aa->conditions = ( int * ) calloc( pae->num_conditions, sizeof( int ) );
	}
	aa->num_conditions = 0;
	for ( i = 0; i < pae->num_conditions; i++ ) {
	  lp = pae->conditions[i].predicate;
	  for ( j = 0; j < garity[lp]; j++ ) {
	    largs[j] = pae->conditions[i].args[j];
	  }
	  adr = fact_adress();
	  if ( !lpos[lp][adr] ) {/* condition not reachable: skip effect */
	    break;
	  }
	  if ( !lneg[lp][adr] ) {/* condition always true: skip it */
	    continue;
	  }
	  aa->conditions[aa->num_conditions++] = lindex[lp][adr];
	}
	if ( i < pae->num_conditions ) {/* found unreachable condition: free condition space */
	  free( aa->conditions );
	  continue;
	}

	/**************************NUMERIC COND*************************/
	if ( pae->num_numeric_conditions > 0 ) {
	  aa->numeric_conditions_comp = ( Comparator * ) 
	    calloc( pae->num_numeric_conditions, sizeof( Comparator ) );
	  aa->numeric_conditions_lh = ( ExpNode_pointer * )
	    calloc( pae->num_numeric_conditions, sizeof( ExpNode_pointer ) );
	  aa->numeric_conditions_rh = ( ExpNode_pointer * )
	    calloc( pae->num_numeric_conditions, sizeof( ExpNode_pointer ) );
	  for ( i = 0; i < pae->num_numeric_conditions; i++ ) {
	    aa->numeric_conditions_lh[i] = NULL;
	    aa->numeric_conditions_rh[i] = NULL;
	  }
	  aa->num_numeric_conditions = 0;
	}
	for ( i = 0; i < pae->num_numeric_conditions; i++ ) {
	  aa->numeric_conditions_comp[aa->num_numeric_conditions] = pae->numeric_conditions_comp[i];
	  aa->numeric_conditions_lh[aa->num_numeric_conditions] = copy_Exp( pae->numeric_conditions_lh[i] );
	  if ( !set_relevants_in_exp( &(aa->numeric_conditions_lh[aa->num_numeric_conditions]) ) ) break;
	  aa->numeric_conditions_rh[aa->num_numeric_conditions] = copy_Exp( pae->numeric_conditions_rh[i] );
	  if ( !set_relevants_in_exp( &(aa->numeric_conditions_rh[aa->num_numeric_conditions]) ) ) break;
	  aa->num_numeric_conditions++;
	}
	if ( i < pae->num_numeric_conditions ) {
	  /* numeric effect uses undefined fluent in condition -->
	   * THROW WHOLE ACTION AWAY! done by breaking out of the 
	   * effects loop, which will be catched below overall
	   * effect handling.
	   */
	  break;
	}
	/**************************NUMERIC COND - END*************************/

	/* now create the add and del effects.
	 */
	if ( pae->num_adds > 0 ) {
	  aa->adds = ( int * ) calloc( pae->num_adds, sizeof( int ) );
	}
	aa->num_adds = 0;
	for ( i = 0; i < pae->num_adds; i++ ) {
	  lp = pae->adds[i].predicate;
	  for ( j = 0; j < garity[lp]; j++ ) {
	    largs[j] = pae->adds[i].args[j];
	  }
	  adr = fact_adress();
	  if ( !lneg[lp][adr] ) {/* effect always true: skip it */
	    continue;
	  }
	  aa->adds[aa->num_adds++] = lindex[lp][adr];
	}

	if ( pae->num_dels > 0 ) {
	  aa->dels = ( int * ) calloc( pae->num_dels, sizeof( int ) );
	}
	aa->num_dels = 0;
	for ( i = 0; i < pae->num_dels; i++ ) {
	  lp = pae->dels[i].predicate;
	  for ( j = 0; j < garity[lp]; j++ ) {
	    largs[j] = pae->dels[i].args[j];
	  }
	  adr = fact_adress();
	  if ( !lpos[lp][adr] ) {/* effect always false: skip it */
	    continue;
	  }
	  aa->dels[aa->num_dels++] = lindex[lp][adr];
	}
	if ( i < pae->num_dels ) break;

	/**************************NUMERIC EFFECTS*************************/
	if ( pae->num_numeric_effects > 0 ) {
	  aa->numeric_effects_neft = ( NumericEffectType * ) 
	    calloc( pae->num_numeric_effects, sizeof( NumericEffectType ) );
	  aa->numeric_effects_fl = ( int * )
	    calloc( pae->num_numeric_effects, sizeof( int ) );
	  aa->numeric_effects_rh = ( ExpNode_pointer * )
	    calloc( pae->num_numeric_effects, sizeof( ExpNode_pointer ) );
	  aa->num_numeric_effects = 0;
	}
	for ( i = 0; i < pae->num_numeric_effects; i++ ) {
	  aa->numeric_effects_neft[aa->num_numeric_effects] = pae->numeric_effects_neft[i];
	  lf = pae->numeric_effects_fluent[i].function;
	  for ( j = 0; j < gf_arity[lf]; j++ ) {
	    lf_args[j] = pae->numeric_effects_fluent[i].args[j];
	    if ( lf_args[j] < 0 ) {
	      printf("\n\nuninstantiated affected fluent in final actions! debug me.\n\n");
	      exit( 1 );
	    }
	  }
	  adr = fluent_adress();
	  /* if it's -1, simply let it in --- if that effect appears, then 
	   * action is illegal, otherwise not.
	   */
	  aa->numeric_effects_fl[i] = lf_index[lf][adr];
	  if ( lf_index[lf][adr] == -1 ) aa->illegal = TRUE;
	  aa->numeric_effects_rh[aa->num_numeric_effects] = copy_Exp( pae->numeric_effects_rh[i] );
	  if ( !set_relevants_in_exp( &(aa->numeric_effects_rh[aa->num_numeric_effects]) ) ) {
	    aa->illegal = TRUE;
	  }
	  if ( aa->illegal &&
	       aa->num_conditions == 0 &&
	       aa->num_numeric_conditions == 0 ) {
	    break;
	  }
	  /* that's it ???????????????? - !!
	   */
	  aa->num_numeric_effects++;
	}
	if ( i < pae->num_numeric_effects ) {
	  /* an unconditional illegal effekt
	   */
	  break;
	}
	/**************************NUMERIC EFFECTS - END*************************/

	/* this effect is OK. go to next one in PseudoAction.
	 */
	a->num_effects++;
	lnum_effects++;
      }
      if ( pae ) {
	/* we get here if one effect was faulty
	 */
	gnum_actions--;
	if ( p ) {
	  p->next = a->next;
	  t = a;
	  a = a->next;
	  t->next = gtrash_actions;
	  gtrash_actions = t;
	} else {
	  gactions = a->next;
	  t = a;
	  a = a->next;
	  t->next = gtrash_actions;
	  gtrash_actions = t;
	}
      } else {
	p = a;
	a = a->next;
      }
      continue;
    }/* end of if clause for PseudoAction */
    /* if action was neither normop, nor pseudo action determined,
     * then it is an artificial action due to disjunctive goal
     * conditions.
     *
     * these are already in final form.
     */
    p = a;
    a = a->next;
  }/* endfor all actions ! */

}



void instantiate_exp_by_action( ExpNode **n, Action *a )

{

  int j, f, k, h;
  Bool ok;

  switch ( (*n)->connective ) {
  case AD:
    instantiate_exp_by_action( &((*n)->leftson), a );
    instantiate_exp_by_action( &((*n)->rightson), a );
    if ( (*n)->leftson->connective != NUMBER ||
	 (*n)->rightson->connective != NUMBER ) {
      break;
    }
    (*n)->connective = NUMBER;
    (*n)->value = (*n)->leftson->value + (*n)->rightson->value;
    free_ExpNode( (*n)->leftson );
    (*n)->leftson = NULL;
    free_ExpNode( (*n)->rightson );
    (*n)->rightson = NULL;
    break;
  case SU:
    instantiate_exp_by_action( &((*n)->leftson), a );
    instantiate_exp_by_action( &((*n)->rightson), a );
    if ( (*n)->leftson->connective != NUMBER ||
	 (*n)->rightson->connective != NUMBER ) {
      break;
    }
    (*n)->connective = NUMBER;
    (*n)->value = (*n)->leftson->value - (*n)->rightson->value;
    free_ExpNode( (*n)->leftson );
    (*n)->leftson = NULL;
    free_ExpNode( (*n)->rightson );
    (*n)->rightson = NULL;
    break;
  case MU:
    instantiate_exp_by_action( &((*n)->leftson), a );
    instantiate_exp_by_action( &((*n)->rightson), a );
    if ( (*n)->leftson->connective != NUMBER ||
	 (*n)->rightson->connective != NUMBER ) {
      break;
    }
    (*n)->connective = NUMBER;
    (*n)->value = (*n)->leftson->value * (*n)->rightson->value;
    free_ExpNode( (*n)->leftson );
    (*n)->leftson = NULL;
    free_ExpNode( (*n)->rightson );
    (*n)->rightson = NULL;
    break;
  case DI:
    instantiate_exp_by_action( &((*n)->leftson), a );
    instantiate_exp_by_action( &((*n)->rightson), a );
    if ( (*n)->leftson->connective != NUMBER ||
	 (*n)->rightson->connective != NUMBER ) {
      break;
    }
    if ( (*n)->rightson->value == 0 ) {
      /* kind of unclean: simply leave that in here;
       * we will later determine the right thing 
       * to do with it.
       */
      break;
    }
    (*n)->connective = NUMBER;
    (*n)->value = (*n)->leftson->value / (*n)->rightson->value;
    free_ExpNode( (*n)->leftson );
    (*n)->leftson = NULL;
    free_ExpNode( (*n)->rightson );
    (*n)->rightson = NULL;
    break;
  case MINUS:
    instantiate_exp_by_action( &((*n)->son), a );
    if ( (*n)->son->connective != NUMBER ) break;
    (*n)->connective = NUMBER;
    (*n)->value = ((float) (-1)) * (*n)->son->value;
    free_ExpNode( (*n)->son );
    (*n)->son = NULL;
    break;    
  case NUMBER:
    break;
  case FHEAD:
    f = (*n)->fluent->function;
    ok = TRUE;
    for ( j = 0; j < gf_arity[f]; j++ ) {
      h = ( (*n)->fluent->args[j] < 0 ) ?
	a->inst_table[DECODE_VAR( (*n)->fluent->args[j] )] : (*n)->fluent->args[j];
      if ( h < 0 ) {
	ok = FALSE;
      } else {
	(*n)->fluent->args[j] = h;
      }
    }
    if ( !ok ) {
      printf("\n\nnon-instantiated fluent in final actiona! debug me!!\n\n");
      exit( 1 );
    }
    if ( gis_changed[f] ) break;
    for ( j = 0; j < gnum_initial_function[f]; j++ ) {
      for ( k = 0; k < gf_arity[f]; k++ ) {
	if ( ginitial_function[f][j].fluent.args[k] !=
	     (*n)->fluent->args[k] ) break;
      }
      if ( k < gf_arity[f] ) continue;
      (*n)->connective = NUMBER;
      (*n)->value = ginitial_function[f][j].value;
      break;
    }
    break;
  default:
    printf("\n\ninst. exp by action: wrong specifier %d",
	   (*n)->connective);
    exit( 1 );
  }

}




















/**************************************************
 * CONNECTIVITY GRAPH. ULTRA CLEAN REPRESENTATION *
 **************************************************/




















void build_connectivity_graph( void )

{

  int i, j, k, l, n_op, n_ef, fl, ef, ef_, m;
  float val;
  Action *a;
  ActionEffect *e;

  gnum_ft_conn = gnum_relevant_facts;
  gnum_fl_conn = gnum_relevant_fluents;
  gnum_op_conn = gnum_actions;
  gft_conn = ( FtConn * ) calloc( gnum_ft_conn, sizeof( FtConn ) );
  gfl_conn = ( FlConn * ) calloc( gnum_fl_conn, sizeof( FlConn ) );
  gop_conn = ( OpConn * ) calloc( gnum_op_conn, sizeof( OpConn ) );
  gef_conn = ( EfConn * ) calloc( lnum_effects, sizeof( EfConn ) );
  gnum_ef_conn = 0;

  for ( i = 0; i < gnum_ft_conn; i++ ) {
    gft_conn[i].num_PC = 0;
    gft_conn[i].num_A = 0;
    gft_conn[i].num_D = 0;

    gft_conn[i].axiom_added = FALSE;

    gft_conn[i].rand = random() % BIG_INT;
  }

  gnum_real_fl_conn = 0;
  for ( i = 0; i < gnum_fl_conn; i++ ) {
    gfl_conn[i].num_PC = 0;
    gfl_conn[i].num_IN = 0;
    gfl_conn[i].num_AS = 0;

    if ( grelevant_fluents_lnf[i] == NULL ) {
      gfl_conn[i].artificial = FALSE;
      gnum_real_fl_conn++;
      gfl_conn[i].rand = random() % BIG_INT;
    } else {
      /* once we're in here we'll stay as all artificial 
       * fluents are appended to the end.
       */
      gfl_conn[i].artificial = TRUE;
      gfl_conn[i].lnf_F = ( int * ) 
	calloc( grelevant_fluents_lnf[i]->num_pF, sizeof( int ) );
      gfl_conn[i].lnf_C = ( float * ) 
	calloc( grelevant_fluents_lnf[i]->num_pF, sizeof( float ) );
      for ( j = 0; j < grelevant_fluents_lnf[i]->num_pF; j++ ) {
	gfl_conn[i].lnf_F[j] = grelevant_fluents_lnf[i]->pF[j];
	gfl_conn[i].lnf_C[j] = grelevant_fluents_lnf[i]->pC[j];
      }
      gfl_conn[i].num_lnf = grelevant_fluents_lnf[i]->num_pF;
    }
  }


  /* why not do this here?
   */
  gmneed_start_D = ( Bool * ) calloc( gnum_real_fl_conn, sizeof( Bool ) );
  gmneed_start_V = ( float * ) calloc( gnum_real_fl_conn, sizeof( float ) );


  for ( i = 0; i < gnum_op_conn; i++ ) {
    gop_conn[i].num_E = 0;
  }

  for ( i = 0; i < lnum_effects; i++ ) {
    gef_conn[i].num_PC = 0;
    gef_conn[i].num_f_PC = 0;
    gef_conn[i].num_A = 0;
    gef_conn[i].num_D = 0;
    gef_conn[i].num_I = 0;
    gef_conn[i].num_IN = 0;
    gef_conn[i].num_AS = 0;

    gef_conn[i].illegal = FALSE;
    gef_conn[i].removed = FALSE;
  }


  /* determine if there are conditional effects.
   */
  gconditional_effects = FALSE;
  for ( a = gactions; a; a = a->next ) {
    for ( i = 0; i < a->num_effects; i++ ) {
      e = &(a->effects[i]);
      if ( e->num_conditions > 0 ) {
	break;
      }
      if ( e->num_lnf_conditions > 0 ) {
	break;
      }
    }
    if ( i < a->num_effects ) break;
  }
  if ( a ) {
    printf("\n\ntask contains conditional effects. turning off state domination.\n\n");
    gconditional_effects = TRUE;
  }

  n_op = 0;
  n_ef = 0;
  for ( a = gactions; a; a = a->next ) {
    gop_conn[n_op].action = a;
    gop_conn[n_op].axiom = a->axiom;
    if ( a->num_effects == 0 ) {
      continue;
    }

    gop_conn[n_op].E = ( int * ) calloc( a->num_effects, sizeof( int ) );
    for ( i = 0; i < a->num_effects; i++ ) {
      e = &(a->effects[i]);
      gef_conn[n_ef].cost = e->cost;
      if ( e->removed ) {
	/* this one disappeared through summarization
	 */
	continue;
      }
      gop_conn[n_op].E[gop_conn[n_op].num_E++] = n_ef;
      gef_conn[n_ef].op = n_op;
      if ( e->illegal ) {
	gef_conn[n_ef].illegal = TRUE;
      }

      /*****************************CONDS********************************/
      gef_conn[n_ef].PC = ( int * ) 
	calloc( e->num_conditions + a->num_preconds, sizeof( int ) );
      for ( j = 0; j < a->num_preconds; j++ ) {
	for ( k = 0; k < gef_conn[n_ef].num_PC; k++ ) {
	  if ( gef_conn[n_ef].PC[k] == a->preconds[j] ) break;
	}
	if ( k < gef_conn[n_ef].num_PC ) continue;
	gef_conn[n_ef].PC[gef_conn[n_ef].num_PC++] = a->preconds[j];
      }
      for ( j = 0; j < e->num_conditions; j++ ) {
	for ( k = 0; k < gef_conn[n_ef].num_PC; k++ ) {
	  if ( gef_conn[n_ef].PC[k] == e->conditions[j] ) break;
	}
	if ( k < gef_conn[n_ef].num_PC ) continue;
	gef_conn[n_ef].PC[gef_conn[n_ef].num_PC++] = e->conditions[j];
      }
      /* similar thing for numeric conditions.
       */
      gef_conn[n_ef].f_PC_comp = ( Comparator * ) 
	calloc( e->num_lnf_conditions + a->num_lnf_preconds, sizeof( Comparator ) );
      gef_conn[n_ef].f_PC_fl = ( int * ) 
	calloc( e->num_lnf_conditions + a->num_lnf_preconds, sizeof( int ) );
      gef_conn[n_ef].f_PC_c = ( float * ) 
	calloc( e->num_lnf_conditions + a->num_lnf_preconds, sizeof( float ) );
      gef_conn[n_ef].f_PC_direct_comp = ( Comparator * ) calloc( gnum_fl_conn, sizeof( Comparator ) );
      for ( j = 0; j < gnum_fl_conn; j++ ) {
	gef_conn[n_ef].f_PC_direct_comp[j] = IGUAL;
      }
      gef_conn[n_ef].f_PC_direct_c = ( float * ) calloc( gnum_fl_conn, sizeof( float ) );
      for ( j = 0; j < a->num_lnf_preconds; j++ ) {
	if ( a->lnf_preconds_lh[j]->num_pF != 1 ) {
	  printf("\n\nnon 1 card. in comp lh final pre copyover.\n\n");
	  exit( 1 );
	}
	for ( k = 0; k < gef_conn[n_ef].num_f_PC; k++ ) {
	  if ( gef_conn[n_ef].f_PC_fl[k] == a->lnf_preconds_lh[j]->pF[0] ) break;
	}
	if ( k < gef_conn[n_ef].num_f_PC ) {
	  if ( a->lnf_preconds_rh[j] < gef_conn[n_ef].f_PC_c[k] ) {
	    /* weaker cond
	     */
	    continue;
	  }
	  if ( a->lnf_preconds_rh[j] > gef_conn[n_ef].f_PC_c[k] ) {
	    /* stronger cond
	     */
	    gef_conn[n_ef].f_PC_c[k] = a->lnf_preconds_rh[j];
	    gef_conn[n_ef].f_PC_comp[k] = a->lnf_preconds_comp[j];
	    continue;
	  }
	  if ( a->lnf_preconds_comp[j] == GE ) {
	    /* we might need to strengthen our comp
	     */
	    gef_conn[n_ef].f_PC_comp[k] = GE;
	  }
	} else {
	  gef_conn[n_ef].f_PC_comp[gef_conn[n_ef].num_f_PC] = a->lnf_preconds_comp[j];
	  gef_conn[n_ef].f_PC_fl[gef_conn[n_ef].num_f_PC] = a->lnf_preconds_lh[j]->pF[0];
	  gef_conn[n_ef].f_PC_c[gef_conn[n_ef].num_f_PC++] = a->lnf_preconds_rh[j];
	}
      }
      for ( j = 0; j < e->num_lnf_conditions; j++ ) {
	if ( e->lnf_conditions_lh[j]->num_pF != 1 ) {
	  printf("\n\nnon 1 card. in comp lh final cond copyover.\n\n");
	  exit( 1 );
	}
	for ( k = 0; k < gef_conn[n_ef].num_f_PC; k++ ) {
	  if ( gef_conn[n_ef].f_PC_fl[k] == e->lnf_conditions_lh[j]->pF[0] ) break;
	}
	if ( k < gef_conn[n_ef].num_f_PC ) {
	  if ( e->lnf_conditions_rh[j] < gef_conn[n_ef].f_PC_c[k] ) {
	    continue;
	  }
	  if ( e->lnf_conditions_rh[j] > gef_conn[n_ef].f_PC_c[k] ) {
	    gef_conn[n_ef].f_PC_c[k] = e->lnf_conditions_rh[j];
	    gef_conn[n_ef].f_PC_comp[k] = e->lnf_conditions_comp[j];
	    continue;
	  }
	  if ( e->lnf_conditions_comp[j] == GE ) {
	    gef_conn[n_ef].f_PC_comp[k] = GE;
	  }
	} else {
	  gef_conn[n_ef].f_PC_comp[gef_conn[n_ef].num_f_PC] = e->lnf_conditions_comp[j];
	  gef_conn[n_ef].f_PC_fl[gef_conn[n_ef].num_f_PC] = e->lnf_conditions_lh[j]->pF[0];
	  gef_conn[n_ef].f_PC_c[gef_conn[n_ef].num_f_PC++] = e->lnf_conditions_rh[j];
	}
      }
      /* now arrange the direct access structures from that.
       */
      for ( j = 0; j < gef_conn[n_ef].num_f_PC; j++ ) {
	gef_conn[n_ef].f_PC_direct_comp[gef_conn[n_ef].f_PC_fl[j]] = gef_conn[n_ef].f_PC_comp[j]; 
	gef_conn[n_ef].f_PC_direct_c[gef_conn[n_ef].f_PC_fl[j]] = gef_conn[n_ef].f_PC_c[j]; 
      }
     /*****************************CONDS - END********************************/


      if ( e->illegal ) {
	/* we don't care about the effects if they're illegal -
	 * all we care about is whether the condition is true or not.
	 */
	n_ef++;
	gnum_ef_conn++;
	continue;
      }
      /*****************************EFFECTS********************************/
      gef_conn[n_ef].A = ( int * ) calloc( e->num_adds, sizeof( int ) );
      gef_conn[n_ef].D = ( int * ) calloc( e->num_dels, sizeof( int ) );
      gef_conn[n_ef].IN_fl = ( int * ) calloc( e->num_lnf_effects, sizeof( int ) );
      gef_conn[n_ef].IN_fl_ = ( int * ) calloc( e->num_lnf_effects, sizeof( int ) );
      gef_conn[n_ef].IN_c = ( float * ) calloc( e->num_lnf_effects, sizeof( float ) );
      gef_conn[n_ef].AS_fl = ( int * ) calloc( e->num_lnf_effects, sizeof( int ) );
      gef_conn[n_ef].AS_fl_ = ( int * ) calloc( e->num_lnf_effects, sizeof( int ) );
      gef_conn[n_ef].AS_c = ( float * ) calloc( e->num_lnf_effects, sizeof( float ) );

      /* duplicates removed in summarize already.
       *
       * but don't include adds that are in the conds.
       * --- those are true anyway.
       *
       * and don't include dels that are in the adds
       * --- those will be re-added anyway.
       *
       * NOTE: it is important that we use the *original* add list
       *       not the already reduced one, for the delete check!
       *       otherwise it may be that a delete that's in the add
       *       and also in the cond stays in!
       *
       *       IT IS ALSO IMPORTANT THAT WE DO BOTH!!!, i.e. if we do 
       *       the ads reduction then we *must* also do the dels 
       *       reduction to avoid that things are deleted that 
       *       would otherwise have been re-added.       
       */
      for ( j = 0; j < e->num_adds; j++ ) {
	for ( k = 0; k < gef_conn[n_ef].num_PC; k++ ) {
	  if ( gef_conn[n_ef].PC[k] == e->adds[j] ) break;
	}
	if ( k < gef_conn[n_ef].num_PC ) continue;
	gef_conn[n_ef].A[gef_conn[n_ef].num_A++] = e->adds[j];
      }
      for ( j = 0; j < e->num_dels; j++ ) {
	for ( k = 0; k < e->num_adds; k++ ) {
	  if ( e->adds[k] == e->dels[j] ) break;
	}
	if ( k < e->num_adds ) continue;
	gef_conn[n_ef].D[gef_conn[n_ef].num_D++] = e->dels[j];
      }

      /* numeric part
       */
      for ( j = 0; j < e->num_lnf_effects; j++ ) {
	if ( e->lnf_effects_neft[j] != INCREASE ) continue;
	gef_conn[n_ef].IN_fl[gef_conn[n_ef].num_IN] = e->lnf_effects_fl[j];
	if ( e->lnf_effects_rh[j]->num_pF == 1 ) {
	  if ( e->lnf_effects_rh[j]->pF[0] < 0 ) {
	    printf("\n\nnon-relevant fluent in final copying to conn.\n\n");
	    exit( 1 );
	  }
	  gef_conn[n_ef].IN_fl_[gef_conn[n_ef].num_IN] = e->lnf_effects_rh[j]->pF[0];
	} else {
	  if ( e->lnf_effects_rh[j]->num_pF != 0 ) {
	    printf("\n\nnon-1 or 0 number of fl_ in copying to conn\n\n");
	    exit( 1 );
	  }
	  gef_conn[n_ef].IN_fl_[gef_conn[n_ef].num_IN] = -1;
	}
	gef_conn[n_ef].IN_c[gef_conn[n_ef].num_IN++] = e->lnf_effects_rh[j]->c;
      }
      /* now remove increasers by nothing.
       */
      j = 0;
      while ( j < gef_conn[n_ef].num_IN ) {
	if ( gef_conn[n_ef].IN_fl_[j] != -1 ||
	     gef_conn[n_ef].IN_c[j] != 0 ) {
	  j++;
	  continue;
	}
	for ( k = j; k < gef_conn[n_ef].num_IN - 1; k++ ) {
	  gef_conn[n_ef].IN_fl[k] = gef_conn[n_ef].IN_fl[k+1];
	  gef_conn[n_ef].IN_fl_[k] = gef_conn[n_ef].IN_fl_[k+1];
	  gef_conn[n_ef].IN_c[k] = gef_conn[n_ef].IN_c[k+1];
	}
	gef_conn[n_ef].num_IN--;
      }
      /* now: the assigners...
       */
      for ( j = 0; j < e->num_lnf_effects; j++ ) {
	if ( e->lnf_effects_neft[j] != ASSIGN ) continue;
	gef_conn[n_ef].AS_fl[gef_conn[n_ef].num_AS] = e->lnf_effects_fl[j];
	if ( e->lnf_effects_rh[j]->num_pF == 1 ) {
	  if ( e->lnf_effects_rh[j]->pF[0] < 0 ) {
	    printf("\n\nnon-relevant fluent in final copying to conn.\n\n");
	    exit( 1 );
	  }
	  gef_conn[n_ef].AS_fl_[gef_conn[n_ef].num_AS] = e->lnf_effects_rh[j]->pF[0];
	} else {
	  if ( e->lnf_effects_rh[j]->num_pF != 0 ) {
	    printf("\n\nnon-1 or 0 number of fl_ in copying to conn\n\n");
	    exit( 1 );
	  }
	  gef_conn[n_ef].AS_fl_[gef_conn[n_ef].num_AS] = -1;
	}
	gef_conn[n_ef].AS_c[gef_conn[n_ef].num_AS++] = e->lnf_effects_rh[j]->c;
      }
      /*****************************EFFECTS - END********************************/
      
      n_ef++;
      gnum_ef_conn++;
    }/* end all a->effects */


    /*****************************EMPTY EFFECTS********************************/
    if ( gop_conn[n_op].num_E >= 1 ) {
      /* CHECK EMPTY EFFECTS!
       *
       * two step process --- first, remove all effects that are entirely empty.
       *                      second, check if all remaining effects are illegal
       *                      or only delete:
       *                      in that case, the op will never do any good so we 
       *                      remove all its effects.
       */
      i = 0;
      while ( i < gop_conn[n_op].num_E ) {
	/* illegal effects *must* stay in!!!
	 */
	if ( gef_conn[gop_conn[n_op].E[i]].illegal ) {
	  i++;
	  continue;
	}
	if ( gef_conn[gop_conn[n_op].E[i]].num_A != 0 ||
	     gef_conn[gop_conn[n_op].E[i]].num_D != 0 ||
	     gef_conn[gop_conn[n_op].E[i]].num_IN != 0 ||
	     gef_conn[gop_conn[n_op].E[i]].num_AS != 0 ) {
	  i++;
	  continue;
	}
	/* we keep it in the gef_conn (seems easier), 
	 * but mark it as removed, which will exclude it from everything.
	 */
	gef_conn[gop_conn[n_op].E[i]].removed = TRUE;
	for ( j = i; j < gop_conn[n_op].num_E - 1; j++ ) {
	  gop_conn[n_op].E[j] = gop_conn[n_op].E[j+1];
	}
	gop_conn[n_op].num_E--;
      }

      m = 0;
      for ( i = 0; i < gop_conn[n_op].num_E; i++ ) {
	if ( gef_conn[gop_conn[n_op].E[i]].illegal ) {
	  m++;
	  continue;
	}
	if ( gef_conn[gop_conn[n_op].E[i]].num_A == 0 &&
	     gef_conn[gop_conn[n_op].E[i]].num_IN == 0 &&
	     gef_conn[gop_conn[n_op].E[i]].num_AS == 0 ) {
	  m++;
	}
      }
      if ( m == gop_conn[n_op].num_E ) {
	/* all remaining effects illegal or solely-deleters.
	 */
	for ( i = 0; i < gop_conn[n_op].num_E; i++ ) {
	  gef_conn[gop_conn[n_op].E[i]].removed = TRUE;
	}
	gop_conn[n_op].num_E = 0;
      }
    }
    /*****************************EMPTY EFFECTS - END********************************/


    /*****************************IMPLIED EFFECTS********************************/
    if ( gop_conn[n_op].num_E > 1 ) {
      for ( i = 0; i < gop_conn[n_op].num_E; i++ ) {
	ef = gop_conn[n_op].E[i];
	gef_conn[ef].I = ( int * ) calloc( gop_conn[n_op].num_E, sizeof( int ) );
	gef_conn[ef].num_I = 0;
      }    
      for ( i = 0; i < gop_conn[n_op].num_E - 1; i++ ) {
	ef = gop_conn[n_op].E[i];
	for ( j = i+1; j < gop_conn[n_op].num_E; j++ ) {
	  ef_ = gop_conn[n_op].E[j];
	  /* ef ==> ef_ ? */
	  for ( k = 0; k < gef_conn[ef_].num_PC; k++ ) {
	    for ( l = 0; l < gef_conn[ef].num_PC; l++ ) {
	      if ( gef_conn[ef].PC[l] == gef_conn[ef_].PC[k] ) break;
	    }
	    if ( l == gef_conn[ef].num_PC ) break;
	  }
	  if ( k == gef_conn[ef_].num_PC ) {
	    for ( k = 0; k < gnum_fl_conn; k++ ) {
	      if ( gef_conn[ef_].f_PC_direct_comp[k] == IGUAL ) continue;
	      if ( gef_conn[ef].f_PC_direct_comp[k] == IGUAL ||
		   gef_conn[ef].f_PC_direct_c[k] < gef_conn[ef_].f_PC_direct_c[k] ||
		   ( gef_conn[ef].f_PC_direct_c[k] == gef_conn[ef_].f_PC_direct_c[k] &&
		     gef_conn[ef].f_PC_direct_comp[k] == GEQ && 
		     gef_conn[ef_].f_PC_direct_comp[k] == GE ) ) break;
	    }
	    if ( k == gnum_fl_conn ) {
	      gef_conn[ef].I[gef_conn[ef].num_I++] = ef_;
	    }
	  }
	  /* ef_ ==> ef ? */
	  for ( k = 0; k < gef_conn[ef].num_PC; k++ ) {
	    for ( l = 0; l < gef_conn[ef_].num_PC; l++ ) {
	      if ( gef_conn[ef_].PC[l] == gef_conn[ef].PC[k] ) break;
	    }
	    if ( l == gef_conn[ef_].num_PC ) break;
	  }
	  if ( k == gef_conn[ef].num_PC ) {
	    for ( k = 0; k < gnum_fl_conn; k++ ) {
	      if ( gef_conn[ef].f_PC_direct_comp[k] == IGUAL ) continue;
	      if ( gef_conn[ef_].f_PC_direct_comp[k] == IGUAL ||
		   gef_conn[ef_].f_PC_direct_c[k] < gef_conn[ef].f_PC_direct_c[k] ||
		   ( gef_conn[ef_].f_PC_direct_c[k] == gef_conn[ef].f_PC_direct_c[k] &&
		     gef_conn[ef_].f_PC_direct_comp[k] == GEQ && 
		     gef_conn[ef].f_PC_direct_comp[k] == GE ) ) break;
	    }
	    if ( k == gnum_fl_conn ) {
	      gef_conn[ef_].I[gef_conn[ef_].num_I++] = ef;
	    }
	  }
	}
      }
    }
    /*****************************IMPLIED EFFECTS - END********************************/

    /* op cost is sum of eff costs + gtt*1:
     * [gtt is multiplicator of TOTAL-TIME in final metric; if no
     * total-time part in metric, it is 0]
     * ie eff-costs plus the cost for the time taken by 1 more step.
     */
    gop_conn[n_op].cost = gtt;
    if ( gop_conn[n_op].num_E > 0 ) {
      for ( i = 0; i < gop_conn[n_op].num_E; i++ ) {
	ef = gop_conn[n_op].E[i];
	if ( gef_conn[ef].illegal ) {
	  continue;
	}
	if ( gef_conn[ef].removed ) {
	  continue;
	}
	gop_conn[n_op].cost += gef_conn[ef].cost;
      }
    }

    /* first sweep: only count the space we need for the fact arrays !
     */
    if ( gop_conn[n_op].num_E > 0 ) {
      for ( i = 0; i < gop_conn[n_op].num_E; i++ ) {
	ef = gop_conn[n_op].E[i];
	for ( j = 0; j < gef_conn[ef].num_PC; j++ ) {
	  gft_conn[gef_conn[ef].PC[j]].num_PC++;
	}
 	for ( j = 0; j < gef_conn[ef].num_A; j++ ) {
	  gft_conn[gef_conn[ef].A[j]].num_A++;
	  if ( gop_conn[n_op].axiom ) {
	      gft_conn[gef_conn[ef].A[j]].axiom_added = TRUE;
	  }
	}
	for ( j = 0; j < gef_conn[ef].num_D; j++ ) {
	  gft_conn[gef_conn[ef].D[j]].num_D++;
	}
	/* similar increments for flconn
	 */
	for ( j = 0; j < gef_conn[ef].num_f_PC; j++ ) {
	  gfl_conn[gef_conn[ef].f_PC_fl[j]].num_PC++;
	}
 	for ( j = 0; j < gef_conn[ef].num_IN; j++ ) {
	  gfl_conn[gef_conn[ef].IN_fl[j]].num_IN++;
	}
 	for ( j = 0; j < gef_conn[ef].num_AS; j++ ) {
	  gfl_conn[gef_conn[ef].AS_fl[j]].num_AS++;
	}
      }
    }
    

    n_op++;
  }

  /*****************************FLCONN********************************/
  for ( i = 0; i < gnum_ft_conn; i++ ) {
    if ( gft_conn[i].num_PC > 0 ) {
      gft_conn[i].PC = ( int * ) calloc( gft_conn[i].num_PC, sizeof( int ) );
    }
    gft_conn[i].num_PC = 0;
    if ( gft_conn[i].num_A > 0 ) {
      gft_conn[i].A = ( int * ) calloc( gft_conn[i].num_A, sizeof( int ) );
    }
    gft_conn[i].num_A = 0;
    if ( gft_conn[i].num_D > 0 ) {
      gft_conn[i].D = ( int * ) calloc( gft_conn[i].num_D, sizeof( int ) );
    }
    gft_conn[i].num_D = 0;
  }
  for ( i = 0; i < gnum_ef_conn; i++ ) {
    if ( gef_conn[i].removed ) continue;
    for ( j = 0; j < gef_conn[i].num_PC; j++ ) {
      gft_conn[gef_conn[i].PC[j]].PC[gft_conn[gef_conn[i].PC[j]].num_PC++] = i;
    }
    for ( j = 0; j < gef_conn[i].num_A; j++ ) {
      gft_conn[gef_conn[i].A[j]].A[gft_conn[gef_conn[i].A[j]].num_A++] = i;
    }
    for ( j = 0; j < gef_conn[i].num_D; j++ ) {
      gft_conn[gef_conn[i].D[j]].D[gft_conn[gef_conn[i].D[j]].num_D++] = i;
    }
  }
  /*****************************FTCONN - END********************************/


  /*****************************FLCONN********************************/
  /* similar thing for flconn
   */
  for ( i = 0; i < gnum_fl_conn; i++ ) {
    if ( gfl_conn[i].num_PC > 0 ) {
      gfl_conn[i].PC = ( int * ) calloc( gfl_conn[i].num_PC, sizeof( int ) );
    }
    gfl_conn[i].num_PC = 0;
    if ( gfl_conn[i].num_IN > 0 ) {
      gfl_conn[i].IN = ( int * ) calloc( gfl_conn[i].num_IN, sizeof( int ) );
      gfl_conn[i].IN_fl_ = ( int * ) calloc( gfl_conn[i].num_IN, sizeof( int ) );
      gfl_conn[i].IN_c = ( float * ) calloc( gfl_conn[i].num_IN, sizeof( float ) );
    }
    gfl_conn[i].num_IN = 0;
    if ( gfl_conn[i].num_AS > 0 ) {
      gfl_conn[i].AS = ( int * ) calloc( gfl_conn[i].num_AS, sizeof( int ) );
      gfl_conn[i].AS_fl_ = ( int * ) calloc( gfl_conn[i].num_AS, sizeof( int ) );
      gfl_conn[i].AS_c = ( float * ) calloc( gfl_conn[i].num_AS, sizeof( float ) );
    }
    gfl_conn[i].num_AS = 0;
  }
  for ( i = 0; i < gnum_ef_conn; i++ ) {
    if ( gef_conn[i].removed ) continue;
    /* PCs
     */
    for ( j = 0; j < gef_conn[i].num_f_PC; j++ ) {
      fl = gef_conn[i].f_PC_fl[j];
      gfl_conn[fl].PC[gfl_conn[fl].num_PC++] = i;
    }
    /* insert increasers by decreasing amount --> 
     * "best" - at least for constant part - are first!
     */
    for ( j = 0; j < gef_conn[i].num_IN; j++ ) {
      fl = gef_conn[i].IN_fl[j];
      val = gef_conn[i].IN_c[j];
      for ( k = 0; k < gfl_conn[fl].num_IN; k++ ) {
	if ( gfl_conn[fl].IN_c[k] < val ) break;
      }
      for ( l = gfl_conn[fl].num_IN; l > k; l-- ) {
	gfl_conn[fl].IN[l] = gfl_conn[fl].IN[l-1];
	gfl_conn[fl].IN_fl_[l] = gfl_conn[fl].IN_fl_[l-1];
	gfl_conn[fl].IN_c[l] = gfl_conn[fl].IN_c[l-1];
      }
      gfl_conn[fl].IN[k] = i;
      gfl_conn[fl].IN_fl_[k] = gef_conn[i].IN_fl_[j];/* the rh fluent */
      gfl_conn[fl].IN_c[k] = val;
      gfl_conn[fl].num_IN++;
    }
    /* insert assigners by decreasing amount --> 
     * "best" - at least for constant part - are first!
     */
    for ( j = 0; j < gef_conn[i].num_AS; j++ ) {
      fl = gef_conn[i].AS_fl[j];
      val = gef_conn[i].AS_c[j];
      for ( k = 0; k < gfl_conn[fl].num_AS; k++ ) {
	if ( gfl_conn[fl].AS_c[k] < val ) break;
      }
      for ( l = gfl_conn[fl].num_AS; l > k; l-- ) {
	gfl_conn[fl].AS[l] = gfl_conn[fl].AS[l-1];
	gfl_conn[fl].AS_fl_[l] = gfl_conn[fl].AS_fl_[l-1];
	gfl_conn[fl].AS_c[l] = gfl_conn[fl].AS_c[l-1];
      }
      gfl_conn[fl].AS[k] = i;
      gfl_conn[fl].AS_fl_[k] = gef_conn[i].AS_fl_[j];/* the rh fluent */
      gfl_conn[fl].AS_c[k] = val;
      gfl_conn[fl].num_AS++;
    }
  }
  /*****************************FLCONN - END********************************/


  /*****************************GOAL********************************/
  gflogic_goal = ( int * ) calloc( gnum_logic_goal, sizeof( int ) );
  for ( j = 0; j < gnum_logic_goal; j++ ) {
    for ( k = 0; k < gnum_flogic_goal; k++ ) {
      if ( gflogic_goal[k] == glogic_goal[j] ) break;
    }
    if ( k < gnum_flogic_goal ) continue;
    gflogic_goal[gnum_flogic_goal++] = glogic_goal[j];
  }
  /* numeric part
   */
  gfnumeric_goal_comp = ( Comparator * ) calloc( gnum_lnf_goal, sizeof( Comparator ) );
  gfnumeric_goal_fl = ( int * ) calloc( gnum_lnf_goal, sizeof( int ) );
  gfnumeric_goal_c = ( float * ) calloc( gnum_lnf_goal, sizeof( float ) );
  for ( j = 0; j < gnum_lnf_goal; j++ ) {
    if ( glnf_goal_lh[j]->num_pF != 1 ) {
      printf("\n\nnon 1 card. in comp lh final goal copyover.\n\n");
      exit( 1 );
    }
    for ( k = 0; k < gnum_fnumeric_goal; k++ ) {
      if ( gfnumeric_goal_fl[k] == glnf_goal_lh[j]->pF[0] ) break;
    }
    if ( k < gnum_fnumeric_goal ) {
      if ( glnf_goal_rh[j] < gfnumeric_goal_c[k] ) continue;
      if ( glnf_goal_rh[j] > gfnumeric_goal_c[k] ) {
	gfnumeric_goal_comp[k] = glnf_goal_comp[j];
	gfnumeric_goal_c[k] = glnf_goal_rh[j];
	continue;
      }
      if ( glnf_goal_comp[j] == GE ) {
	gfnumeric_goal_comp[k] = GE;
      }
    } else {
      gfnumeric_goal_comp[gnum_fnumeric_goal] = glnf_goal_comp[j];
      gfnumeric_goal_fl[gnum_fnumeric_goal] = glnf_goal_lh[j]->pF[0];
      gfnumeric_goal_c[gnum_fnumeric_goal++] = glnf_goal_rh[j];
    }
  }
  gfnumeric_goal_direct_comp = ( Comparator * ) calloc( gnum_fl_conn, sizeof( Comparator ) );
  for ( j = 0; j < gnum_fl_conn; j++ ) {
    gfnumeric_goal_direct_comp[j] = IGUAL;
  }
  gfnumeric_goal_direct_c = ( float * ) calloc( gnum_fl_conn, sizeof( float ) );
  for ( k = 0; k < gnum_fnumeric_goal; k++ ) {
    gfnumeric_goal_direct_comp[gfnumeric_goal_fl[k]] = gfnumeric_goal_comp[k];
    gfnumeric_goal_direct_c[gfnumeric_goal_fl[k]] = gfnumeric_goal_c[k];
  }
  /*****************************GOAL - END********************************/



  /********************
   * safety: if there are numeric precs/goals, need to turn 
   * cost-minimizing rplans off!!!
   * (see comments with def of gcost_rplans
   */
  for ( i = 0; i < gnum_ef_conn; i++ ) {
    if ( gcost_rplans && gef_conn[i].num_f_PC > 0 ) {
      printf("\nwarning: numeric precondition. turning cost-minimizing relaxed plans OFF.");
      gcost_rplans = FALSE;
      break;
    }
  }
  if ( gcost_rplans && gnum_fnumeric_goal > 0 ) {
    printf("\nwarning: numeric goal. turning cost-minimizing relaxed plans OFF.");
    gcost_rplans = FALSE;
  }




  if ( gcmd_line.display_info == 125 ) {
    printf("\n\ncreated connectivity graph as follows:");

    printf("\n\n------------------OP ARRAY:-----------------------");
    for ( i = 0; i < gnum_op_conn; i++ ) {
      printf("\n\nOP %d: ", i);
      if ( gop_conn[i].axiom ) printf("(axiom) ");
      print_op_name( i );
      printf(" cost %f", gop_conn[i].cost);
      printf("\n----------EFFS:");
      for ( j = 0; j < gop_conn[i].num_E; j++ ) {
	printf("\neffect %d", gop_conn[i].E[j]);
      }
    }
    
    printf("\n\n-------------------EFFECT ARRAY:----------------------");
    for ( i = 0; i < gnum_ef_conn; i++ ) {
      printf("\n\neffect %d of op %d cost %f: ", i, gef_conn[i].op, gef_conn[i].cost);
      print_op_name( gef_conn[i].op );
      if ( gef_conn[i].illegal ) printf(" ******ILLEGAL************************");
      if ( gef_conn[i].removed ) printf(" ******REMOVED************************");
      printf("\n----------PCS:");
      for ( j = 0; j < gef_conn[i].num_PC; j++ ) {
	printf("\n");
	print_ft_name( gef_conn[i].PC[j] );
      }
      printf("\n----------f_PCS:");
      for ( j = 0; j < gef_conn[i].num_f_PC; j++ ) {
	printf("\n");
	print_fl_name( gef_conn[i].f_PC_fl[j] );
	if ( gef_conn[i].f_PC_comp[j] == GEQ ) {
	  printf(" >= ");
	} else {
	  printf(" > ");
	}
	printf("%f", gef_conn[i].f_PC_c[j]);
      }
      printf("\nDIRECT: ");
      for ( j = 0; j < gnum_fl_conn; j++ ) {
	if ( gef_conn[i].f_PC_direct_comp[j] == IGUAL ) {
	  printf("IGUAL  |  ");
	}
	if ( gef_conn[i].f_PC_direct_comp[j] == GEQ ) {
	  printf(">= %f  |  ", gef_conn[i].f_PC_direct_c[j]);
	}
	if ( gef_conn[i].f_PC_direct_comp[j] == GE ) {
	  printf("> %f  |  ", gef_conn[i].f_PC_direct_c[j]);
	}
      }
      if ( gef_conn[i].illegal ) continue;
      printf("\n----------ADDS:");
      for ( j = 0; j < gef_conn[i].num_A; j++ ) {
	printf("\n");
	print_ft_name( gef_conn[i].A[j] );
      }
      printf("\n----------DELS:");
      for ( j = 0; j < gef_conn[i].num_D; j++ ) {
	printf("\n");
	print_ft_name( gef_conn[i].D[j] );
      }
      printf("\n----------INCREASE:");
      for ( j = 0; j < gef_conn[i].num_IN; j++ ) {
	printf("\n");
	print_fl_name( gef_conn[i].IN_fl[j] );
	printf(" by ");
	if ( gef_conn[i].IN_fl_[j] >= 0 ) {
	  print_fl_name( gef_conn[i].IN_fl_[j] );
	  printf(" + %f", gef_conn[i].IN_c[j]);
	} else {
	  printf("%f", gef_conn[i].IN_c[j]);
	}
      }
      printf("\n----------ASSIGN:");
      for ( j = 0; j < gef_conn[i].num_AS; j++ ) {
	printf("\n");
	print_fl_name( gef_conn[i].AS_fl[j] );
	printf(" to ");
	if ( gef_conn[i].AS_fl_[j] >= 0 ) {
	  print_fl_name( gef_conn[i].AS_fl_[j] );
	  printf(" + %f", gef_conn[i].AS_c[j]);
	} else {
	  printf("%f", gef_conn[i].AS_c[j]);
	}
      }
      printf("\n----------IMPLIEDS:");
      for ( j = 0; j < gef_conn[i].num_I; j++ ) {
	printf("\nimplied effect %d of op %d: ", 
	       gef_conn[i].I[j], gef_conn[gef_conn[i].I[j]].op);
	print_op_name( gef_conn[gef_conn[i].I[j]].op );
      }
    }
    
    printf("\n\n----------------------FT ARRAY:-----------------------------");
    for ( i = 0; i < gnum_ft_conn; i++ ) {
      printf("\n\nFT: ");
      print_ft_name( i );
      printf(" rand: %d", gft_conn[i].rand);
      printf(" --------- AXIOM ADDED %d", gft_conn[i].axiom_added);
      printf("\n----------PRE COND OF:");
      for ( j = 0; j < gft_conn[i].num_PC; j++ ) {
	printf("\neffect %d", gft_conn[i].PC[j]);
	printf(" - op "); print_op_name( gef_conn[gft_conn[i].PC[j]].op );
      }
      printf("\n----------ADD BY:");
      for ( j = 0; j < gft_conn[i].num_A; j++ ) {
	printf("\neffect %d", gft_conn[i].A[j]);
	printf(" - op "); print_op_name( gef_conn[gft_conn[i].A[j]].op );
      }
      printf("\n----------DEL BY:");
      for ( j = 0; j < gft_conn[i].num_D; j++ ) {
	printf("\neffect %d", gft_conn[i].D[j]);
	printf(" - op "); print_op_name( gef_conn[gft_conn[i].D[j]].op );
      }
    }
    
    printf("\n\n----------------------FLUENT ARRAY:-----------------------------");
    for ( i = 0; i < gnum_fl_conn; i++ ) {
      printf("\n\nFL: ");
      print_fl_name( i );
      printf("\n----------PRE COND OF:");
      for ( j = 0; j < gfl_conn[i].num_PC; j++ ) {
	printf("\neffect %d", gfl_conn[i].PC[j]);
	printf(" - op "); print_op_name( gef_conn[gfl_conn[i].PC[j]].op );
      }
      printf("\n----------INCREASED BY:");
      for ( j = 0; j < gfl_conn[i].num_IN; j++ ) {
	if ( gfl_conn[i].IN_fl_[j] == -1 ) {
	  printf("\neffect %d  ---  %f", gfl_conn[i].IN[j], gfl_conn[i].IN_c[j]);
	  printf("  ---  op "); print_op_name( gef_conn[gfl_conn[i].IN[j]].op );
	} else {
	  printf("\neffect %d  ---  ", gfl_conn[i].IN[j]);
	  print_fl_name( gfl_conn[i].IN_fl_[j] );
	  printf(" + %f", gfl_conn[i].IN_c[j]);
	  printf("  ---  op "); print_op_name( gef_conn[gfl_conn[i].IN[j]].op );
	}
      }
      printf("\n----------ASSIGNED BY:");
      for ( j = 0; j < gfl_conn[i].num_AS; j++ ) {
	if ( gfl_conn[i].AS_fl_[j] == -1 ) {
	  printf("\neffect %d  ---  %f", gfl_conn[i].AS[j], gfl_conn[i].AS_c[j]);
	  printf("  ---  op "); print_op_name( gef_conn[gfl_conn[i].AS[j]].op );
	} else {
	  printf("\neffect %d  ---  ", gfl_conn[i].AS[j]);
	  print_fl_name( gfl_conn[i].AS_fl_[j] );
	  printf(" + %f", gfl_conn[i].AS_c[j]);
	  printf("  ---  op "); print_op_name( gef_conn[gfl_conn[i].AS[j]].op );
	}
      }
      if ( gfl_conn[i].artificial ) {
	printf("\n----------ARTIFICIAL FOR:");
	for ( j = 0; j < gfl_conn[i].num_lnf; j++ ) {
	  printf(" %f*", gfl_conn[i].lnf_C[j]);
	  print_fl_name( gfl_conn[i].lnf_F[j] );
	  if ( j < gfl_conn[i].num_lnf - 1 ) {
	    printf(" +");
	  }
	}
      } else {
	printf("\n----------REAL");
      }
    }

    printf("\n\n----------------------GOAL:-----------------------------");
    for ( j = 0; j < gnum_flogic_goal; j++ ) {
      printf("\n");
      print_ft_name( gflogic_goal[j] );
    }
    for ( j = 0; j < gnum_fnumeric_goal; j++ ) {
      printf("\n");
      print_fl_name( gfnumeric_goal_fl[j] );
      if ( gfnumeric_goal_comp[j] == GEQ ) {
	printf(" >= ");
      } else {
	printf(" > ");
      }
      printf("%f", gfnumeric_goal_c[j]);
    }
    printf("\nDIRECT: ");
    for ( j = 0; j < gnum_fl_conn; j++ ) {
      if ( gfnumeric_goal_direct_comp[j] == IGUAL ) {
	printf("IGUAL  |  ");
      }
      if ( gfnumeric_goal_direct_comp[j] == GEQ ) {
	printf(">= %f  |  ", gfnumeric_goal_direct_c[j]);
      }
      if ( gfnumeric_goal_direct_comp[j] == GE ) {
	printf("> %f  |  ", gfnumeric_goal_direct_c[j]);
      }
    }
    
    printf("\n\n");
  }
    
}



