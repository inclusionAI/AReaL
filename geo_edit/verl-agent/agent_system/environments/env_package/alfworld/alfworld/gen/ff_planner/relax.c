
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
 * File: relax.c
 * Description: this file handles the relaxed planning problem, i.e.,
 *              the code is responsible for the heuristic evaluation
 *              of states during search.
 *
 *              --- THE HEART PEACE OF THE FF PLANNER ! ---
 *
 *              here: linear tasks +=,-=,:= / le / le <comp> le
 *
 *
 * Author: Joerg Hoffmann 2000--2002, 2011
 *
 *********************************************************************/ 









#include "ff.h"

#include "output.h"
#include "memory.h"

#include "expressions.h"

#include "relax.h"
#include "search.h"







/* local globals
 */








/* fixpoint
 */
int *lF;
int lnum_F;
int *lE;
int lnum_E;

int *lch_E;
int lnum_ch_E;

int *l0P_E;
int lnum_0P_E;





/* 1P extraction
 */
int **lgoals_at;
int *lnum_goals_at;

float **lf_goals_c_at;
Comparator **lf_goals_comp_at;

int lh;

int *lch_F;
int lnum_ch_F;

int *lused_O;
int lnum_used_O;

int *lin_plan_E;
int lnum_in_plan_E;


/* helpful actions numerical helpers
 */
Comparator *lHcomp;
float *lHc;




















/*************************************
 * helper, for -1 == INFINITY method *
 *************************************/












Bool LESS( int a, int b )

{

  if ( a == INFINITY ) {
    return FALSE;
  }

  if ( b == INFINITY ) {
    return TRUE;
  }

  return ( a < b ? TRUE : FALSE );

}



Bool FLOAT_LE( float a, float b )

{

  if ( b == INFINITY ) {
    return TRUE;
  }

  if ( a == INFINITY ) {
    return FALSE;
  }

  return ( a <= b ? TRUE : FALSE );

}














/***********************************
 * FUNCTIONS ACCESSED FROM OUTSIDE *
 ***********************************/

















int get_1P( State *S )

{

  int max, h;
  Bool solvable;

  gevaluated_states++;

  solvable = build_fixpoint( S, &max );

  if ( gcmd_line.display_info == 126 ) {
    print_fixpoint_result();
  }

  if ( solvable ) {
    h = extract_1P( max );
  } else {
    h = INFINITY;
  }

  reset_fixpoint( max );

  return h;

}



int get_1P_and_H( State *S )

{

  int max, h;
  Bool solvable;

  gevaluated_states++;

  solvable = build_fixpoint( S, &max );

  if ( gcmd_line.display_info == 126 ) {
    print_fixpoint_result();
  }

  if ( solvable ) {
    h = extract_1P( max );
    collect_H_info();
  } else {
    h = INFINITY;
  }

  reset_fixpoint( max );

  return h;

}



int get_1P_and_A( State *S )

{

  int max, h;
  Bool solvable;

  gevaluated_states++;

  solvable = build_fixpoint( S, &max );

  if ( gcmd_line.display_info == 126 ) {
    print_fixpoint_result();
  }

  if ( solvable ) {
    h = extract_1P( max );
  } else {
    h = INFINITY;
  }

  collect_1P_and_A_info();
  reset_fixpoint( max );

  return h;

}



void collect_1P_and_A_info( void )

{

  static Bool first_call = TRUE;

  int i;

  if ( first_call ) {
    gA = ( int * ) calloc( gnum_op_conn, sizeof( int ) );
    gnum_A = 0;
    first_call = FALSE;
  }

  if ( gcmd_line.debug ) {
    printf("\ncollect_1P_and_A_info");
  }

  for ( i = 0; i < gnum_A; i++ ) {
    gop_conn[gA[i]].is_in_A = FALSE;
  }
  gnum_A = 0;

  for ( i = 0; i < lnum_E; i++ ) {
    if ( gef_conn[lE[i]].level != 0 ) break;
    if ( gcmd_line.debug ) {
      printf("\ngot applicable op: ");
      print_op_name(gef_conn[lE[i]].op);
    }
    if ( gop_conn[gef_conn[lE[i]].op].is_in_A ) {
      if ( gcmd_line.debug ) {
	printf(" -- already in, skipping it!");
      }
      continue;
    }
    if ( gop_conn[gef_conn[lE[i]].op].axiom ) {
      if ( gcmd_line.debug ) {
	printf(" -- axiom, skipping it!");
      }
      continue;
    }
    if ( gcmd_line.debug ) {
      printf(" -- adding it!");
    }
    gop_conn[gef_conn[lE[i]].op].is_in_A = TRUE;
    gA[gnum_A++] = gef_conn[lE[i]].op;
  }

}



void get_A( State *S )

{

  int i;

  initialize_fixpoint( S );
  
  for ( i = 0; i < lnum_F; i++ ) {
    activate_ft( lF[i], 0 );
  }
  for ( i = 0; i < lnum_0P_E; i++ ) {
    if ( gef_conn[l0P_E[i]].in_E ) {
      continue;
    }
    new_ef( l0P_E[i] );
  }
  for ( i = 0; i < gnum_fl_conn; i++ ) {
    activate_fl( i, 0 );
  }

  collect_A_info();

  /* 0 should be enough here...
   */
  reset_fixpoint( 1 );

}



void collect_A_info( void )

{

  static Bool first_call = TRUE;

  int i;

  if ( first_call ) {
    gA = ( int * ) calloc( gnum_op_conn, sizeof( int ) );
    gnum_A = 0;
    first_call = FALSE;
  }

  if ( gcmd_line.debug ) {
    printf("\ncollect_A_info");
  }

  for ( i = 0; i < gnum_A; i++ ) {
    gop_conn[gA[i]].is_in_A = FALSE;
  }
  gnum_A = 0;

  for ( i = 0; i < lnum_E; i++ ) {
    /* levels are not set unless we actually build the RPG!
/*     if ( gef_conn[lE[i]].level != 0 ) break; */
    if ( gcmd_line.debug ) {
      printf("\ngot applicable op: ");
      print_op_name(gef_conn[lE[i]].op);
    }
    if ( gop_conn[gef_conn[lE[i]].op].is_in_A ) {
      if ( gcmd_line.debug ) {
	printf(" -- already in, skipping it!");
      }
      continue;
    }
    if ( gop_conn[gef_conn[lE[i]].op].axiom ) {
      if ( gcmd_line.debug ) {
	printf(" -- axiom, skipping it!");
      }
      continue;
    }
    if ( gcmd_line.debug ) {
      printf(" -- adding it!");
    }
    gop_conn[gef_conn[lE[i]].op].is_in_A = TRUE;
    gA[gnum_A++] = gef_conn[lE[i]].op;
  }

}



void get_A_axioms( State *S )

{

  int i;

  initialize_fixpoint( S );
  
  for ( i = 0; i < lnum_F; i++ ) {
    activate_ft( lF[i], 0 );
  }
  for ( i = 0; i < lnum_0P_E; i++ ) {
    if ( gef_conn[l0P_E[i]].in_E ) {
      continue;
    }
    new_ef( l0P_E[i] );
  }
  for ( i = 0; i < gnum_fl_conn; i++ ) {
    activate_fl( i, 0 );
  }

  collect_A_axioms_info();

  /* 0 should be enough here...
   */
  reset_fixpoint( 1 );

}



void collect_A_axioms_info( void )

{

  static Bool first_call = TRUE;

  int i;

  if ( first_call ) {
    gA_axioms = ( int * ) calloc( gnum_op_conn, sizeof( int ) );
    gnum_A_axioms = 0;
    first_call = FALSE;
  }

  if ( gcmd_line.debug ) {
    printf("\ncollect_A_axioms_info");
  }

  for ( i = 0; i < gnum_A_axioms; i++ ) {
    gop_conn[gA_axioms[i]].is_in_A_axioms = FALSE;
  }
  gnum_A_axioms = 0;

  for ( i = 0; i < lnum_E; i++ ) {
    /* levels are not set unless we actually build the RPG!
/*     if ( gef_conn[lE[i]].level != 0 ) break; */
    if ( gcmd_line.debug ) {
      printf("\ngot applicable op: ");
      print_op_name(gef_conn[lE[i]].op);
    }
    if ( gop_conn[gef_conn[lE[i]].op].is_in_A_axioms ) {
      if ( gcmd_line.debug ) {
	printf(" -- already in, skipping it!");
      }
      continue;
    }
    if ( !gop_conn[gef_conn[lE[i]].op].axiom ) {
      if ( gcmd_line.debug ) {
	printf(" -- no axiom, skipping it!");
      }
      continue;
    }
    if ( gcmd_line.debug ) {
      printf(" -- adding it!");
    }

    gop_conn[gef_conn[lE[i]].op].is_in_A_axioms = TRUE;
    gA_axioms[gnum_A_axioms++] = gef_conn[lE[i]].op;
  }

}



























/*******************************
 * RELAXED FIXPOINT ON A STATE *
 *******************************/




































Bool build_fixpoint( State *S, int *max )

{

  int start_ft, stop_ft, start_ef, stop_ef, i, time = 0;
  float costlevel;

  initialize_fixpoint( S );

  start_ft = 0;
  start_ef = 0;
  while ( TRUE ) {
    if ( gcmd_line.debug ) {
      printf("\n======================================FP time %d", time);
    }

    if ( all_goals_activated( time ) ) {
      break;
    }
    if ( time > 0 || lnum_0P_E == 0 ) {
      if ( start_ft == lnum_F ) {
	if ( fluents_hopeless( time ) ) {
	  /* fixpoint, goals not reached
	   */
	  *max = time;
	  return FALSE;
	}
      }
    }
    /* make space if necessary, and copy over
     * info from time to time+1 for fluents
     */
    extend_fluent_levels( time );
    for ( i = 0; i < gnum_fl_conn; i++ ) {
      if ( gfl_conn[i].def[time] ) {
	gfl_conn[i].def[time+1] = TRUE;
	gfl_conn[i].level[time+1] = gfl_conn[i].level[time];
      }
    }

    /* determine the next effect layer:
     * - activate the facts
     * - if level 0 activate the no preconds-ops
     * - activate the fluents at their <time> level
     */

    if ( !gcost_rplans ) {
      stop_ft = lnum_F;
    } else {
      /* first, find the latest index of facts whose cost 
       * is the same as that of leading fact
       */
      costlevel = gft_conn[lF[start_ft]].RPGcost;
      for ( i = start_ft; i < lnum_F; i++ ) {
	if ( gft_conn[lF[i]].RPGcost > costlevel ) {
	  break;
	}
      }
      stop_ft = i;
    }

    for ( i = start_ft; i < stop_ft; i++ ) {
      activate_ft( lF[i], time );
    }
    if ( time == 0 ) {
      for ( i = 0; i < lnum_0P_E; i++ ) {
	if ( gef_conn[l0P_E[i]].in_E ) {
	  continue;
	}
	new_ef( l0P_E[i] );
      }
    }
    for ( i = 0; i < gnum_fl_conn; i++ ) {
      activate_fl( i, time );
    }

    /* now say what the benefits of applying the new
     * effect layer are:
     * 
     * the facts,
     * plus the new fluent levels at time + 1.
     */
    stop_ef = lnum_E;
    for ( i = start_ef; i < stop_ef; i++ ) {
      activate_ef( lE[i], time );
    }
    /* on top of what the new effects have added to the fluents, all effects 
     * strictly below <time> can be applied again. Their effect might be 
     * different from what they've done before, as it might depend on
     * the rh fluent fl_
     */
    if ( time > 0 ) {
      for ( i = 0; i < start_ef; i++ ) {
	apply_ef( lE[i], time );
      }
    }
    /* now check whether there became any assigner applicable that is
     * mightier than all the increasers put together.
     *
     * this needs only be done for the non-artificial
     * fluents as the others obviously don't get
     * assigned at all.
     */
    for ( i = 0; i < gnum_real_fl_conn; i++ ) {
      if ( !gfl_conn[i].curr_assigned ) {
	/* no assigner in yet
	 */
	continue;
      }
      if ( !gfl_conn[i].def[time] ) {
	gfl_conn[i].def[time+1] = TRUE;
	gfl_conn[i].level[time+1] = gfl_conn[i].curr_max_assigned;
	continue;
      }
      if ( gfl_conn[i].curr_max_assigned > gfl_conn[i].level[time+1] ) {
	gfl_conn[i].level[time+1] = gfl_conn[i].curr_max_assigned;
      }
    }
    /* finally, determine the new levels of the artificial fluents
     * at the new time step.
     */
    determine_artificial_fl_levels( time + 1 );

    start_ft = stop_ft;
    start_ef = stop_ef;
    time++;
  }

  *max = time;
  return TRUE;

}



Bool fluents_hopeless( int time )

{

  int i, j;
  Bool minusinfty;
  float mneed;

  /* if no real fluent has improved from the previous step to this one,
   * then - with facts not improving either - the process has reached a fixpoint.
   *
   * for the formal details of this, ie. the mneed values, see the JAIR article.
   */

  if ( time == 0 ) {
    /* nothing has happened yet...
     */
    return FALSE;
  }

  for ( i = 0; i < gnum_real_fl_conn; i++ ) {
    gmneed_start_D[i] = gfl_conn[i].def[time-1];
    gmneed_start_V[i] = gfl_conn[i].level[time-1];
  }

  for ( i = 0; i < gnum_real_fl_conn; i++ ) {
    if ( !gfl_conn[i].def[time-1] &&
	 !gfl_conn[i].def[time] ) {
      /* this one has obviously not been made any better
       */
      continue;
    }
    if ( gfl_conn[i].def[time-1] &&
	 gfl_conn[i].level[time] == gfl_conn[i].level[time-1] ) {
      /* this one has not improved either
       */
      continue;
    }
    get_mneed( i, &minusinfty, &mneed );
    if ( gcmd_line.display_info == 333 ) {
      printf("\nstart values:");
      for ( j = 0; j < gnum_real_fl_conn; j++ ) {
	printf("\n"); print_fl_name( j ); 
	printf(" --- %d, %f", gmneed_start_D[j], gmneed_start_V[j]);
      }
      printf("\nmneed "); print_fl_name( i ); 
      printf(" --- %d, %f", minusinfty, mneed);
    }

    if ( minusinfty ) {
      /* this one is not needed at all
       */
      continue;
    }
    if ( gfl_conn[i].def[time-1] && gfl_conn[i].level[time-1] > mneed ) {
      /* here we already had a sufficient value at the last layer.
       */
      continue;
    }
    return FALSE;
  }

  return TRUE;
    
}



void initialize_fixpoint( State *S )

{

  static Bool first_call = TRUE;

  int i;

  if ( first_call ) {
    /* make initial space for fluent levels
     */
    extend_fluent_levels( -1 );

    /* get memory for local globals
     */
    lF = ( int * ) calloc( gnum_ft_conn, sizeof( int ) );
    lE = ( int * ) calloc( gnum_ef_conn, sizeof( int ) );
    lch_E = ( int * ) calloc( gnum_ef_conn, sizeof( int ) );
    l0P_E = ( int * ) calloc( gnum_ef_conn, sizeof( int ) );
    
    /* initialize connectivity graph members for
     * relaxed planning
     */
    lnum_0P_E = 0;
    for ( i = 0; i < gnum_ef_conn; i++ ) {      
      gef_conn[i].level = INFINITY;    
      gef_conn[i].RPGcost = INFINITY; 
   
      gef_conn[i].in_E = FALSE;
      gef_conn[i].num_active_PCs = 0;
      gef_conn[i].ch = FALSE;

      gef_conn[i].num_active_f_PCs = 0;
      
      if ( gef_conn[i].num_PC == 0 &&
	   gef_conn[i].num_f_PC == 0 &&
	   !gef_conn[i].illegal &&
	   !gef_conn[i].removed ) {
	l0P_E[lnum_0P_E++] = i;
      }
    }
    for ( i = 0; i < gnum_op_conn; i++ ) {      
      gop_conn[i].is_in_A = FALSE;
      gop_conn[i].is_in_A_axioms = FALSE;
      gop_conn[i].is_in_H = FALSE;
    }
    for ( i = 0; i < gnum_ft_conn; i++ ) {
      gft_conn[i].level = INFINITY;
      gft_conn[i].RPGcost = INFINITY;

      gft_conn[i].in_F = FALSE;
    }
    for ( i = 0; i < gnum_fl_conn; i++ ) {
      gfl_conn[i].curr_assigned = FALSE;
    }
    first_call = FALSE;
  }

  ghmax = -1;

  lnum_E = 0;
  lnum_ch_E = 0;

  lnum_F = 0;
  for ( i = 0; i < S->num_F; i++ ) {
    if ( gft_conn[S->F[i]].in_F ) {
      continue;
    }
    new_fact( S->F[i], 0 );
  }
  /* only the real fls are ever in there
   */
  for ( i = 0; i < gnum_real_fl_conn; i++ ) {
    if ( !S->f_D[i] ) {
      continue;
    }
    gfl_conn[i].def[0] = TRUE;
    gfl_conn[i].level[0] = S->f_V[i];
  }
  /* now set the art. values from that.
   */
  determine_artificial_fl_levels( 0 );

}



void determine_artificial_fl_levels( int time )

{

  int i, j;
  float l;

  /* for all art. fls
   */
  for ( i = gnum_real_fl_conn; i < gnum_fl_conn; i++ ) {
    l = 0;
    for ( j = 0; j < gfl_conn[i].num_lnf; j++ ) {
      if ( !gfl_conn[gfl_conn[i].lnf_F[j]].def[time] ) break;
      l += (gfl_conn[i].lnf_C[j] * gfl_conn[gfl_conn[i].lnf_F[j]].level[time]);
    }
    if ( j < gfl_conn[i].num_lnf ) {
      /* one part yet undefined.
       */
      continue;
    } else {
      gfl_conn[i].def[time] = TRUE;
      gfl_conn[i].level[time] = l;
    }
  }

}



void extend_fluent_levels( int time )

{

  static int highest_seen;

  Bool *b;
  float *f1;

  int i, j;

  if ( time == -1 ) {
    highest_seen = RELAXED_STEPS_DEFAULT;
    for ( i = 0; i < gnum_fl_conn; i++ ) {
      gfl_conn[i].def = ( Bool * ) calloc( highest_seen, sizeof( Bool ) );
      for ( j = 0; j < highest_seen; j++ ) {
	gfl_conn[i].def[j] = FALSE;
      }
      gfl_conn[i].level = ( float * ) calloc( highest_seen, sizeof( float ) );
    }
    return;
  }

  if ( time + 1 < highest_seen ) return;

  b = ( Bool * ) calloc( time + 1, sizeof( Bool ) );
  f1 = ( float * ) calloc( time + 1, sizeof( float ) );

  highest_seen = time + 10;
  for ( i = 0; i < gnum_fl_conn; i++ ) {
    for ( j = 0; j <= time; j++ ) {
      b[j] = gfl_conn[i].def[j];
      f1[j] = gfl_conn[i].level[j];
    }

    free( gfl_conn[i].def );
    free( gfl_conn[i].level );
    gfl_conn[i].def = ( Bool * ) calloc( highest_seen, sizeof( Bool ) );
    gfl_conn[i].level = ( float * ) calloc( highest_seen, sizeof( float ) );

    for ( j = 0; j <= time; j++ ) {
      gfl_conn[i].def[j] = b[j];
      gfl_conn[i].level[j] = f1[j];
    }
    for ( j = time + 1; j < highest_seen; j++ ) {
      gfl_conn[i].def[j] = FALSE;
    }
  }

  free( b );
  free( f1 );

}



void activate_ft( int index, int time )

{

  int i;

  if ( gcmd_line.debug ) {
    printf("\nactivate fact time %d: ", time);
    print_ft_name(index);
  }

  gft_conn[index].level = time;

  for ( i = 0; i < gft_conn[index].num_PC; i++ ) {
    /* never activate illegal effects.
     */
    if ( gef_conn[gft_conn[index].PC[i]].illegal ) continue;

    if ( gcmd_line.debug ) {
      printf("\nincrementing activePC of ");
      print_op_name(gef_conn[gft_conn[index].PC[i]].op);
      printf(" from %d to %d; total: %d", 
	     gef_conn[gft_conn[index].PC[i]].num_active_PCs,
	     gef_conn[gft_conn[index].PC[i]].num_active_PCs+1,
	     gef_conn[gft_conn[index].PC[i]].num_PC);
    }
    
    gef_conn[gft_conn[index].PC[i]].num_active_PCs++;
    if ( !gef_conn[gft_conn[index].PC[i]].ch ) {
      gef_conn[gft_conn[index].PC[i]].ch = TRUE;
      lch_E[lnum_ch_E++] = gft_conn[index].PC[i];
    }
    if ( gef_conn[gft_conn[index].PC[i]].num_active_PCs ==
	 gef_conn[gft_conn[index].PC[i]].num_PC &&
	 gef_conn[gft_conn[index].PC[i]].num_active_f_PCs ==
	 gef_conn[gft_conn[index].PC[i]].num_f_PC ) {
      if ( gcmd_line.debug ) {
	printf("\ncalling new ef at time %d for ft ", time);
	print_ft_name(index);
      }
      new_ef( gft_conn[index].PC[i] );
    }
  }

}



void activate_fl( int index, int time )

{

  int i, ef;

  if ( !gfl_conn[index].def[time] ) return;

  for ( i = 0; i < gfl_conn[index].num_PC; i++ ) {
    ef = gfl_conn[index].PC[i];
    if ( gef_conn[ef].illegal ) continue;

    if ( gef_conn[ef].f_PC_direct_comp[index] == IGUAL ) {
      printf("\n\nprec of addressed ef does not care about fl!\n\n");
      exit( 1 );
    }
    if ( !number_comparison_holds( gef_conn[ef].f_PC_direct_comp[index],
				   gfl_conn[index].level[time],
				   gef_conn[ef].f_PC_direct_c[index] ) ) {
      /* the level is not yet high enough for this one
       */
      continue;
    }
    if ( time > 0 &&
	 gfl_conn[index].def[time-1] &&
	 number_comparison_holds( gef_conn[ef].f_PC_direct_comp[index],
				  gfl_conn[index].level[time-1],
				  gef_conn[ef].f_PC_direct_c[index] ) ) {
      /* last time this was already in: that one's old!
       * do not count it!
       */
      continue;
    }
    gef_conn[ef].num_active_f_PCs++;
    if ( !gef_conn[ef].ch ) {
      gef_conn[ef].ch = TRUE;
      lch_E[lnum_ch_E++] = ef;
    }
    if ( gef_conn[ef].num_active_PCs == gef_conn[ef].num_PC &&
	 gef_conn[ef].num_active_f_PCs == gef_conn[ef].num_f_PC ) {
      if ( gcmd_line.debug ) {
	printf("\ncalling new ef at time %d for fluent ", time);
	print_fl_name(index);
      }
      new_ef( ef );
    }
  }

}



void activate_ef( int index, int time )

{

  int i, fl;
  float val;

/*   if ( gef_conn[index].removed ) { */
/*     printf("\n\nactivating removed effect!!\n\n"); */
/*     exit( 1 ); */
/*   } */
  if ( gef_conn[index].illegal ) {
    printf("\n\nactivating illegal effect!!\n\n");
    exit( 1 );
  }

  gef_conn[index].level = time;

  if ( gcmd_line.debug ) {
    printf("\nactivate effect time %d: ", time);
    print_op_name(gef_conn[index].op);
    printf(" with own cost %f and RPGcost %f", 
	   gop_conn[gef_conn[index].op].cost,
	   gef_conn[index].RPGcost);
   }

  for ( i = 0; i < gef_conn[index].num_A; i++ ) {
    if ( gft_conn[gef_conn[index].A[i]].in_F ) {
      continue;
    }
    if ( gcmd_line.debug ) {
      printf("\ncalling new fact at time %d for op ", time);
      print_op_name(gef_conn[index].op);
      printf(" with own cost %f and RPGcost %f", 
	     gop_conn[gef_conn[index].op].cost,
	     gef_conn[index].RPGcost);
    }
    new_fact( gef_conn[index].A[i], gef_conn[index].RPGcost );
  }

  for ( i = 0; i < gef_conn[index].num_IN; i++ ) {
    fl = gef_conn[index].IN_fl[i];
    if ( !gfl_conn[fl].def[time] ) {
      /* in principle, we could skip this action here as
       * it affects a fluent that is yet undefined.
       *
       * ...seems difficult to integrate into implementation,
       * does not matter much probably (?): so we relax
       * the task further in the sense that we allow this action
       * here, only it has of course no effect on the fluent.
       */
      continue;
    }
    /* value is either the constant, or the (artificial lnf?) fl_ level
     * at <time> plus the constant.
     */
    if ( gef_conn[index].IN_fl_[i] == -1 ) {
      val = gef_conn[index].IN_c[i];
    } else {
      if ( !gfl_conn[gef_conn[index].IN_fl_[i]].def[time] ) {
	/* this one does not help us here.
	 */
	continue;
      }
      val = gfl_conn[gef_conn[index].IN_fl_[i]].level[time] + gef_conn[index].IN_c[i];
    }
    /* we only consider the effect if it helps us.
     */
    if ( val > 0 ) {
      gfl_conn[fl].level[time+1] += val;
    }
  }

  /* the assigners are remembered in a parallel sort of way...
   */
  for ( i = 0; i < gef_conn[index].num_AS; i++ ) {
    fl = gef_conn[index].AS_fl[i];
    if ( gef_conn[index].AS_fl_[i] == -1 ) {
      val = gef_conn[index].AS_c[i];
    } else {
      if ( !gfl_conn[gef_conn[index].AS_fl_[i]].def[time] ) {
	/* this one does not help us here.
	 */
	continue;
      }
      val = gfl_conn[gef_conn[index].AS_fl_[i]].level[time] + gef_conn[index].AS_c[i];
    }
    if ( gfl_conn[fl].curr_assigned ) {
      if ( gfl_conn[fl].curr_max_assigned < val ) {
	gfl_conn[fl].curr_max_assigned = val;
      }
    } else {
      gfl_conn[fl].curr_assigned = TRUE;
      gfl_conn[fl].curr_max_assigned = val;
    }
  }

}



/* this one is used to apply effects already there
 * at later time steps - necessary because their
 * numeric effects might have changed.
 */
void apply_ef( int index, int time )

{

  int i, fl;
  float val;

/*   if ( gef_conn[index].removed ) { */
/*     printf("\n\napplying removed effect!!\n\n"); */
/*     exit( 1 ); */
/*   } */
  if ( gef_conn[index].illegal ) {
    return;
  }

  /* only numerical effects matter.
   */
  for ( i = 0; i < gef_conn[index].num_IN; i++ ) {
    fl = gef_conn[index].IN_fl[i];
    if ( !gfl_conn[fl].def[time] ) {
      continue;
    }
    if ( gef_conn[index].IN_fl_[i] == -1 ) {
      val = gef_conn[index].IN_c[i];
    } else {
      if ( !gfl_conn[gef_conn[index].IN_fl_[i]].def[time] ) {
	/* no effect here.
	 */
	continue;
      }
      val = gfl_conn[gef_conn[index].IN_fl_[i]].level[time] + gef_conn[index].IN_c[i];
    }
    if ( val > 0 ) {
      gfl_conn[fl].level[time+1] += val;
    }
  }

  for ( i = 0; i < gef_conn[index].num_AS; i++ ) {
    fl = gef_conn[index].AS_fl[i];
    if ( gef_conn[index].AS_fl_[i] == -1 ) {
      val = gef_conn[index].AS_c[i];
    } else {
      if ( !gfl_conn[gef_conn[index].AS_fl_[i]].def[time] ) {
	/* no effect here.
	 */
	continue;
      }
      val = gfl_conn[gef_conn[index].AS_fl_[i]].level[time] + gef_conn[index].AS_c[i];
    }
    if ( gfl_conn[fl].curr_assigned ) {
      if ( gfl_conn[fl].curr_max_assigned < val ) {
	gfl_conn[fl].curr_max_assigned = val;
      }
    } else {
      gfl_conn[fl].curr_assigned = TRUE;
      gfl_conn[fl].curr_max_assigned = val;
    }
  }

}



void new_fact( int index, float RPGcost )

{

  int i, j, N;

  if ( gcmd_line.debug ) {
    printf("\nnew fact cost %f: ", RPGcost);
    print_ft_name(index);
  }

  if ( !gcost_rplans ) {
    if ( gft_conn[index].in_F ) {
      return;
    }
    lF[lnum_F++] = index;
    gft_conn[index].in_F = TRUE;
    return;
  }

  /* if the fact is already here, then we may decrease its cost
   * and move it further to the front. thus, find its current position j
   * and remember it as the "right end" N of the sequence to be considered.
   */
  if ( gft_conn[index].in_F ) {
    if ( gft_conn[index].RPGcost <= RPGcost ) {
      return;
    }

    for ( j = 0; j < lnum_F; j++ ) {
      if ( lF[j] == index ) break;
    }
    if ( j == lnum_F ) {
      printf("\n\nDEBUG ME: didn't find already *in* fact\n\n");
      exit(1);
    }
    N = j;
  } else {
    gft_conn[index].in_F = TRUE;
    N = lnum_F;
    lnum_F++;
  }

  gft_conn[index].RPGcost = RPGcost;
  /* find the position i where ft needs to be inserted
   * IMPORTANT: insert new fact AFTER previous facts with
   * same value. Otherwise, the old facts will end up within
   * the part of the lF list to be activated in the next round 
   */
  for ( i = 0; i < N; i++ ) {
    if ( RPGcost < gft_conn[lF[i]].RPGcost ) break;
  }
  for ( j = N; j > i; j-- ) {
    lF[j] = lF[j-1];
  }
  lF[i] = index;

}



void new_ef( int index )

{

  int i;
  float max;

  if ( gcmd_line.debug ) {
    printf("\nnew effect for op ");
    print_op_name(gef_conn[index].op);
  }

  if ( gef_conn[index].in_E ) {
    printf("\n\nDEBUG ME: new effect already in\n");
    exit(1);
  }

  lE[lnum_E++] = index;
  gef_conn[index].in_E = TRUE;

  if ( !gcost_rplans ) {
    return;
  }

  max = 0;
  if ( gcmd_line.debug ) {
    printf("... precond costs: ");
  }
  for ( i = 0; i < gef_conn[index].num_PC; i++ ) {
    if ( gft_conn[gef_conn[index].PC[i]].RPGcost < 0 ) {
      printf("\n\nDEBUG ME: PC fact of new ef has cost %f? fact ", 
	     gft_conn[gef_conn[index].PC[i]].RPGcost);
      print_ft_name(gef_conn[index].PC[i]);
      printf(" with level %d of op ", gft_conn[gef_conn[index].PC[i]].level); 
      print_op_name(gef_conn[index].op);
      printf("\n");
      exit( 1 );
    }
    if ( gcmd_line.debug ) {
      printf("%f, ", gft_conn[gef_conn[index].PC[i]].RPGcost);
    }
    if ( max < gft_conn[gef_conn[index].PC[i]].RPGcost ) {
      max = gft_conn[gef_conn[index].PC[i]].RPGcost;
    }
  }
  gef_conn[index].RPGcost = max + gop_conn[gef_conn[index].op].cost;
    if ( gcmd_line.debug ) {
      printf("own cost %f, total cost %f", 
	     gop_conn[gef_conn[index].op].cost,
	     gef_conn[index].RPGcost);
    }

}



void reset_fixpoint( int max )

{

  int i, j;

  for ( i = 0; i < lnum_F; i++ ) {
    gft_conn[lF[i]].level = INFINITY;
    gft_conn[lF[i]].RPGcost = INFINITY;
    gft_conn[lF[i]].in_F = FALSE;
  }
  if ( gcmd_line.debug ) {
    for ( i = 0; i < gnum_ft_conn; i++ ) {
      if ( gft_conn[i].level != INFINITY ) {
	printf("\nDEBUG ME: ft level not properly reset\n");
	exit( 1 );
      } 
      if ( gft_conn[i].RPGcost != INFINITY ) {
	printf("\nDEBUG ME: ft RPGcost not properly reset\n");
	exit( 1 );
      } 
      if ( gft_conn[i].in_F ) {
	printf("\nDEBUG ME: ft in_F not properly reset\n");
	exit( 1 );
      } 
    }
  }

  for ( i = 0; i < lnum_E; i++ ) {
    gef_conn[lE[i]].level = INFINITY;
    gef_conn[lE[i]].RPGcost = INFINITY;
    gef_conn[lE[i]].in_E = FALSE;
  }
  for ( i = 0; i < lnum_ch_E; i++ ) {
    gef_conn[lch_E[i]].num_active_PCs = 0;
    gef_conn[lch_E[i]].num_active_f_PCs = 0;
    gef_conn[lch_E[i]].ch = FALSE;
  }
  if ( gcmd_line.debug ) {
    for ( i = 0; i < gnum_ef_conn; i++ ) {
      if ( gef_conn[i].level != INFINITY ) {
	printf("\nDEBUG ME: ef level not properly reset\n");
	exit( 1 );
      } 
      if ( gef_conn[i].RPGcost != INFINITY ) {
	printf("\nDEBUG ME: ef RPGcost not properly reset\n");
	exit( 1 );
      } 
      if ( gef_conn[i].in_E ) {
	printf("\nDEBUG ME: ef in_E not properly reset\n");
	exit( 1 );
      } 
      if ( gef_conn[i].num_active_PCs != 0 ) {
	printf("\nDEBUG ME: ef active PCs not properly reset\n");
	exit( 1 );
      } 
    }
  }
    
  for ( i = 0; i < gnum_fl_conn; i++ ) {
    for ( j = 0; j <= max; j++ ) {
      gfl_conn[i].def[j] = FALSE;
    }
    gfl_conn[i].curr_assigned = FALSE;
  }

}



Bool all_goals_activated( int time ) 

{

  int i;

  if ( gcost_rplans ) {
    ghmax = -1;
  }
  for ( i = 0; i < gnum_flogic_goal; i++ ) {
    if ( !gft_conn[gflogic_goal[i]].in_F ) {
      return FALSE;
    }
    if ( gcost_rplans ) {
      if ( ghmax == -1 || gft_conn[gflogic_goal[i]].RPGcost > ghmax ) {
	ghmax = gft_conn[gflogic_goal[i]].RPGcost;
      }
    }
  }

  for ( i = 0; i < gnum_fnumeric_goal; i++ ) {
    if ( !gfl_conn[gfnumeric_goal_fl[i]].def[time] ) {
      return FALSE;
    }
    if ( !number_comparison_holds( gfnumeric_goal_comp[i],
				   gfl_conn[gfnumeric_goal_fl[i]].level[time],
				   gfnumeric_goal_c[i] ) ) {
      return FALSE;
    }
  }

  for ( i = 0; i < gnum_flogic_goal; i++ ) {
    if ( gft_conn[gflogic_goal[i]].level == INFINITY ) {
      gft_conn[gflogic_goal[i]].level = time;
    }
  }

  return TRUE;

}



void print_fixpoint_result( void )

{

  int time, i;
  Bool hit, hit_F, hit_E, hit_FL;

  time = 0;
  while ( 1 ) {
    hit = FALSE;
    hit_F = FALSE;
    hit_E = FALSE;
    hit_FL = FALSE;
    for ( i = 0; i < gnum_ft_conn; i++ ) {
      if ( gft_conn[i].level == time ) {
	hit = TRUE;
	hit_F = TRUE;
	break;
      }
    }
    for ( i = 0; i < gnum_ef_conn; i++ ) {
      if ( gef_conn[i].level == time ) {
	hit = TRUE;
	hit_E = TRUE;
	break;
      }
    }
    for ( i = 0; i < gnum_fl_conn; i++ ) {
      if ( gfl_conn[i].def[time] ) {
	hit = TRUE;
	hit_FL = TRUE;
	break;
      }
    }
    if ( !hit ) {
      break;
    }
 
    printf("\n\nLEVEL %d:", time);
    if ( hit_F ) {
      printf("\n\nFACTS:");
      for ( i = 0; i < gnum_ft_conn; i++ ) {
	if ( gft_conn[i].level == time ) {
	  printf("\n");
	  print_ft_name( i );
	  if ( gcost_rplans ) {
	    printf(", cost %f", gft_conn[i].RPGcost);
	  }
	}
      }
    }
    if ( hit_FL ) {
      printf("\n\nFLUENTS:");
      for ( i = 0; i < gnum_fl_conn; i++ ) {
	if ( gfl_conn[i].def[time] ) {
	  printf("\n");
	  print_fl_name( i );
	  printf(": %f", gfl_conn[i].level[time]);
	} else {
	  printf("\n");
	  print_fl_name( i );
	  printf(": UNDEF");
	}
      }
    }
    if ( hit_E ) {
      printf("\n\nEFS:");
      for ( i = 0; i < gnum_ef_conn; i++ ) {
	if ( gef_conn[i].level == time ) {
	  printf("\neffect %d to ", i);
	  print_op_name( gef_conn[i].op );
	  if ( gcost_rplans ) {
	    printf(", cost %f own op-cost %f", gef_conn[i].RPGcost, gop_conn[gef_conn[i].op].cost);
	  }
	}
      }
    }

    time++;
  }

  printf("\n");
  fflush( stdout );

}
 


/* naive implementation of the mneed computation. speedups
 * should be possible by utilizing effective data access, and
 * dynamic programming according to topological sorting wrp to
 * := dependencies.
 * 
 * wonder whether that process is time relevant at all?
 *
 * function is recursive, and will terminate because the :=
 * dependencies are cycle-free.
 */
void get_mneed( int fl, Bool *minusinfty, float *val )

{

  int ef, pc, i, fl_, c, in, g, as;
  float val_, mneed_;
  Bool minusinfty_;

  /* this counts the number of hits; 0 --> mneed is -infty
   */
  c = 0;
  /* check through the actions, i.e. effects. I suppose this could be made
   * *a lot* faster by checking through the PC information of the fl;
   * additionally, we'd need fast access to the art. fluents that fl 
   * participates in.
   */
  for ( ef = 0; ef < gnum_ef_conn; ef++ ) {
    /* the preconds must be supported above their required value.
     */
    for ( pc = 0;  pc < gef_conn[ef].num_f_PC; pc++ ) {
      /* constraint here is gef_conn[ef].f_PC_fl[pc] >= [>] gef_conn[ef].f_PC_c[pc]
       * where lh side can be lnf expression.
       */
      fl_ = gef_conn[ef].f_PC_fl[pc];
      if ( fl_ < 0 ) continue;
      if ( fl_ == fl ) {
	c++;
	val_ = gef_conn[ef].f_PC_c[pc];
	if ( c == 1 || val_ > *val ) {
	  *val = val_;
	}
	continue;
      }
      if ( !gfl_conn[fl_].artificial ) continue;
      for ( i = 0; i < gfl_conn[fl_].num_lnf; i++ ) {
	if ( gfl_conn[fl_].lnf_F[i] == fl ) {
	  /* might be that the expression can't be supported --
	   * if one of the fluents is undefined.
	   */
	  if ( supv( &val_, fl, fl_, gef_conn[ef].f_PC_c[pc] ) ) {
	    c++;
	    if ( c == 1 || val_ > *val ) {
	      *val = val_;
	    }
	  }
	  break;
	}
      }
    }

    /* the += effs must be supported above 0.
     */
    for ( in = 0; in < gef_conn[ef].num_IN; in++ ) {
      /* += eff here is gef_conn[ef].IN_fl[in] += gef_conn[ef].IN_fl_[in] + 
       *                                          gef_conn[ef].IN_c[in]
       * where gef_conn[ef].IN_fl_[in] can be lnf expression.
       */
      /* relevance...
       */
      if ( !gfl_conn[gef_conn[ef].IN_fl[in]].relevant ) {
	continue;
      }
      fl_ = gef_conn[ef].IN_fl_[in];
      if ( fl_ < 0 ) continue;
      if ( fl_ == fl ) {
	c++;
	val_ = (-1) * gef_conn[ef].IN_c[in];
	if ( c == 1 || val_ > *val ) {
	  *val = val_;
	}
	continue;
      }
      if ( !gfl_conn[fl_].artificial ) continue;
      for ( i = 0; i < gfl_conn[fl_].num_lnf; i++ ) {
	if ( gfl_conn[fl_].lnf_F[i] == fl ) {
	  if ( supv( &val_, fl, fl_, (-1) * gef_conn[ef].IN_c[in] ) ) {
	    c++;
	    if ( c == 1 || val_ > *val ) {
	      *val = val_;
	    }
	  }
	  break;
	}
      }
    }

    /* the := effs must be supported above the mneed value of the affected
     * varuable..
     */
    for ( as = 0; as < gef_conn[ef].num_AS; as++ ) {
      /* := eff here is gef_conn[ef].AS_fl[as] := gef_conn[ef].AS_fl_[as] + 
       *                                          gef_conn[ef].AS_c[as]
       * where gef_conn[ef].AS_fl_[as] can be lnf expression.
       */
      /* relevance...
       */
      if ( !gfl_conn[gef_conn[ef].AS_fl[as]].relevant ) {
	continue;
      }
      /* after this, -infty handling is actually superfluous... or so I'd suppose...
       */
      fl_ = gef_conn[ef].AS_fl_[as];
      if ( fl_ < 0 ) continue;
      if ( fl_ == fl ) {
	get_mneed( gef_conn[ef].AS_fl[as], &minusinfty_, &mneed_ );
	if ( !minusinfty_ ) {
	  c++;
	  val_ = mneed_ - gef_conn[ef].AS_c[as];
	  if ( c == 1 || val_ > *val ) {
	    *val = val_;
	  }
	}
	continue;
      }
      if ( !gfl_conn[fl_].artificial ) continue;
      for ( i = 0; i < gfl_conn[fl_].num_lnf; i++ ) {
	if ( gfl_conn[fl_].lnf_F[i] == fl ) {
	  get_mneed( gef_conn[ef].AS_fl[as], &minusinfty_, &mneed_ );
	  if ( !minusinfty_ ) {
	    if ( supv( &val_, fl, fl_, mneed_ - gef_conn[ef].AS_c[as] ) ) {
	      c++;
	      if ( c == 1 || val_ > *val ) {
		*val = val_;
	      }
	    }
	    break;
	  }
	}
      }
    }

  }/* end of ef loop */

  /* check through the numerical goals.
   */
  for ( g = 0; g < gnum_fnumeric_goal; g++ ) {
    /* constraint here is gfnumeric_goal_fl[g] >= [>] gfnumeric_goal_c[g]
     * where lh side can be lnf expression.
     */
    fl_ = gfnumeric_goal_fl[g];
    if ( fl_ < 0 ) continue;
    if ( fl_ == fl ) {
      c++;
      val_ = gfnumeric_goal_c[g];
      if ( c == 1 || val_ > *val ) {
	*val = val_;
      }
    }
    if ( !gfl_conn[fl_].artificial ) continue;
    for ( i = 0; i < gfl_conn[fl_].num_lnf; i++ ) {
      if ( gfl_conn[fl_].lnf_F[i] == fl ) {
	if ( supv( &val_, fl, fl_, gfnumeric_goal_c[g] ) ) {
	  c++;
	  if ( c == 1 || val_ > *val ) {
	    *val = val_;
	  }
	}
	break;
      }
    }
  }

  *minusinfty = ( c == 0 ) ? TRUE : FALSE;

}   



Bool supv( float *val, int fl, int expr, float c_ )

{

  int i, pos = 0;

  *val = c_;

  for ( i = 0; i < gfl_conn[expr].num_lnf; i++ ) {
    if ( gfl_conn[expr].lnf_F[i] == fl ) {
      pos = i;
      continue;
    }
    if ( !gmneed_start_D[gfl_conn[expr].lnf_F[i]] ) {
      /* this expression contains an undef fluent -->
       * no matter how much we increase fl, it won't help
       * at all.
       */
      return FALSE;
    }
    *val -= gfl_conn[expr].lnf_C[i] * 
      gmneed_start_V[gfl_conn[expr].lnf_F[i]];
  }

  /* this here is > 0 (hopefully -- yes, checked that)
   */
  *val /= gfl_conn[expr].lnf_C[pos];

  return TRUE;

}




















/**************************************
 * FIRST RELAXED PLAN (1P) EXTRACTION *
 **************************************/



























int extract_1P( int max )

{

  static Bool first_call = TRUE;
  int i, max_goal_level, time;

  if ( first_call ) {
    for ( i = 0; i < gnum_ft_conn; i++ ) {
      gft_conn[i].is_true = INFINITY;
      gft_conn[i].is_goal = FALSE;
      gft_conn[i].ch = FALSE;
    }
    for ( i = 0; i < gnum_op_conn; i++ ) {
      gop_conn[i].is_used = INFINITY;
    }
    for ( i = 0; i < gnum_ef_conn; i++ ) {
      gef_conn[i].in_plan = INFINITY;
    }
    lch_F = ( int * ) calloc( gnum_ft_conn, sizeof( int ) );
    lnum_ch_F = 0;
    lused_O = ( int * ) calloc( gnum_op_conn, sizeof( int ) );
    lnum_used_O = 0;
    lin_plan_E = ( int * ) calloc( gnum_ef_conn, sizeof( int ) );
    lnum_in_plan_E = 0;
    first_call = FALSE;
  }

  reset_search_info();

  max_goal_level = initialize_goals( max );

  if ( gcost_minimizing ) {
    /* we are optimizing cost;
     * initialize global cost of the relaxed plan.
     *
     * NOTE that this is used for any cost-minimizing search,
     * regardless of whether or not we also use cost-minimizing 
     * rplans!!
     */
    gh_cost = 0;
  }

  lh = 0;
  for ( time = max_goal_level; time > 0; time-- ) {
    achieve_goals( time );
  }

  return lh;

}



int initialize_goals( int max )

{

  static Bool first_call = TRUE;
  static int highest_seen;

  int i, j, max_goal_level, ft, fl;
  Comparator comp;
  float val;

  if ( first_call ) {
    lgoals_at = ( int ** ) calloc( RELAXED_STEPS_DEFAULT, sizeof( int * ) );
    lnum_goals_at = ( int * ) calloc( RELAXED_STEPS_DEFAULT, sizeof( int ) );
    lf_goals_c_at = ( float ** ) calloc( RELAXED_STEPS_DEFAULT, sizeof( float * ) );
    lf_goals_comp_at = ( Comparator ** ) calloc( RELAXED_STEPS_DEFAULT, sizeof( Comparator * ) );
    for ( i = 0; i < RELAXED_STEPS_DEFAULT; i++ ) {
      lgoals_at[i] = ( int * ) calloc( gnum_ft_conn, sizeof( int ) );
      lf_goals_c_at[i] = ( float * ) calloc( gnum_fl_conn, sizeof( float ) );
      lf_goals_comp_at[i] = ( Comparator * ) calloc( gnum_fl_conn, sizeof( Comparator ) );
    }
    highest_seen = RELAXED_STEPS_DEFAULT;

    lHcomp = ( Comparator * ) calloc( gnum_fl_conn, sizeof( Comparator ) );
    lHc = ( float * ) calloc( gnum_fl_conn, sizeof( float ) );
    first_call = FALSE;
  }

  if ( max + 1 > highest_seen ) {
    for ( i = 0; i < highest_seen; i++ ) {
      free( lgoals_at[i] );
      free( lf_goals_c_at[i] );
      free( lf_goals_comp_at[i] );
    }
    free( lgoals_at );
    free( lnum_goals_at );
    free( lf_goals_c_at );
    free( lf_goals_comp_at );
    highest_seen = max + 10;
    lgoals_at = ( int ** ) calloc( highest_seen, sizeof( int * ) );
    lnum_goals_at = ( int * ) calloc( highest_seen, sizeof( int ) );
    lf_goals_c_at = ( float ** ) calloc( highest_seen, sizeof( float * ) );
    lf_goals_comp_at = ( Comparator ** ) calloc( highest_seen, sizeof( Comparator * ) );
    for ( i = 0; i < highest_seen; i++ ) {
      lgoals_at[i] = ( int * ) calloc( gnum_ft_conn, sizeof( int ) );
      lf_goals_c_at[i] = ( float * ) calloc( gnum_fl_conn, sizeof( float ) );
      lf_goals_comp_at[i] = ( Comparator * ) calloc( gnum_fl_conn, sizeof( Comparator ) );
    }
  }

  for ( i = 0; i < max + 1; i++ ) {
    lnum_goals_at[i] = 0;
    for ( j = 0; j < gnum_fl_conn; j++ ) {
      lf_goals_comp_at[i][j] = IGUAL;
      /* probably not necessary; igual...
       */
      lHcomp[j] = IGUAL;
    }
  }

  max_goal_level = 0;
  for ( i = 0; i < gnum_flogic_goal; i++ ) {
    ft = gflogic_goal[i];
    /* level can't be INFINITY as otherwise 1P wouldn't have
     * been called.
     */
    if ( gft_conn[ft].level > max_goal_level ) {
      max_goal_level = gft_conn[ft].level;
    }
    lgoals_at[gft_conn[ft].level][lnum_goals_at[gft_conn[ft].level]++] = ft;
    gft_conn[ft].is_goal = TRUE;
    if ( !gft_conn[ft].ch ) {
      lch_F[lnum_ch_F++] = ft;
      gft_conn[ft].ch = TRUE;
    }
  }

  for ( i = 0; i < gnum_fnumeric_goal; i++ ) {
    fl = gfnumeric_goal_fl[i];
    val = gfnumeric_goal_c[i];
    comp = gfnumeric_goal_comp[i];
    for ( j = 0; j <= max; j++ ) {
      if ( !gfl_conn[fl].def[j] ) continue;
      if ( number_comparison_holds( comp, gfl_conn[fl].level[j], val ) ) break;
    }
    if ( j > max ) {
      printf("\n\nnumeric goal not reached in 1P??\n\n");
      exit( 1 );
    }
    if ( j > max_goal_level ) {
      max_goal_level = j;
    }
    lf_goals_c_at[j][fl] = val;
    lf_goals_comp_at[j][fl] = comp;
  }

  return max_goal_level;

}



void achieve_goals( int time )

{

  int i, j, k, ft, min_e, ef, min_p, p;
  float val;
  float preRPGcost, min_preRPGcost;
  int preRPGlevel, min_preRPGlevel, min_ownlevel;

  /* achieve the goals set at level time >= 1
   */
  
  if ( gcmd_line.display_info == 127 ) {
    printf("\nselecting at step %3d: ", time-1);
  }

  /* before we start, we must translate the artificial goals
   * into real goals.
   */
  for ( i = gnum_real_fl_conn; i < gnum_fl_conn; i++ ) {
    if ( lf_goals_comp_at[time][i] == IGUAL ) {
      /* this one isn't needed
       */
      continue;
    }
    enforce_artificial_goal( i, time );
  }

  /* for helpful actions:
   * remember at time 1 what the goals were.
   */
  if ( time == 1 ) {
    for ( i = 0; i < gnum_real_fl_conn; i++ ) {
      lHcomp[i] = lf_goals_comp_at[time][i];
      lHc[i] = lf_goals_c_at[time][i];
    }
  }

  /* first, push the numeric goals at this level so far down that
   * the requirement for each of them can be fulfilled in the previous
   * level.
   */
  for ( i = 0; i < gnum_real_fl_conn; i++ ) {
    if ( lf_goals_comp_at[time][i] == IGUAL ) {
      continue;
    }
    if ( gfl_conn[i].def[time-1] &&
	 number_comparison_holds( lf_goals_comp_at[time][i],
				  gfl_conn[i].level[time-1],
				  lf_goals_c_at[time][i] ) ) {
      /* this can be solved one step earlier.
       * propagate it downwards and mark as OK.
       */
      update_f_goal( i, time-1, lf_goals_comp_at[time][i], lf_goals_c_at[time][i] );
      lf_goals_comp_at[time][i] = IGUAL;
      continue;
    }
    /* if there is a good assigner, then take it.
     */
    for ( j = 0; j < gfl_conn[i].num_AS; j++ ) {
      ef = gfl_conn[i].AS[j];
      if ( !LESS( gef_conn[ef].level, time ) ) {
	/* we allow any effect that's already there
	 */
	continue;
      }
      if ( gfl_conn[i].AS_fl_[j] != -1 &&
	   !gfl_conn[gfl_conn[i].AS_fl_[j]].def[time-1] ) {
	/* accesses an undefined value.
	 */
	continue;
      }
      if ( gfl_conn[i].AS_fl_[j] != -1 ) {
	val = gfl_conn[gfl_conn[i].AS_fl_[j]].level[time-1] + gfl_conn[i].AS_c[j];
      } else {
	val = gfl_conn[i].AS_c[j];
      }
      if ( !number_comparison_holds( lf_goals_comp_at[time][i],
				     val,
				     lf_goals_c_at[time][i] ) ) {
	/* that one is not strong enough.
	 */
	continue;
      }
      break;
    }
    if ( j < gfl_conn[i].num_AS ) {
      /* ef is an assigner that is strong enough and already there.
       */
      if ( gef_conn[ef].in_plan == time - 1 ) {
	printf("\n\nassigner already selected, nevertheless goal still there\n\n");
	exit( 1 );
      } else {
	if ( gef_conn[ef].in_plan == INFINITY ) {
	  lin_plan_E[lnum_in_plan_E++] = ef;
	}
	gef_conn[ef].in_plan = time - 1;
      }
      /* now select the resp. op at this level, if necessary
       */
      select_op( time, gef_conn[ef].op );
      /* now mark the benefits of that effect, introducing
       * also the fl_ level enforcement goals for each effect
       * that is useful for solving a goal at time: in particular,
       * this will be the one we have just selected.
       */      
      introduce_benefits_and_enforcements( time, ef );
      /* now introduce the new goals
       */
      introduce_pc_goals( time, ef );

      /* care about next fluent
       */
      continue;
    }
    /* debug...
     */
    if ( !gfl_conn[i].def[time-1] ) {
      printf("\n\nall assignerss applied yet goal not fulfilled - undefined below.\n\n");
      exit( 1 );
    }



    /* no good assigner available. thus, push the goal at this level so far 
     * down that its requirement can be fulfilled in the previous level.
     */
    for ( j = 0; j < gfl_conn[i].num_IN; j++ ) {
      /* go through increasers in constant quantity order top to
       * bottom (see inst_final.c);
       */
      ef = gfl_conn[i].IN[j];
      if ( !LESS( gef_conn[ef].level, time ) ) {
	continue;
      }
      if ( gfl_conn[i].IN_fl_[j] != -1 &&
	   !gfl_conn[gfl_conn[i].IN_fl_[j]].def[time-1] ) {
	/* accesses an undefined fluent.
	 */
	continue;
      }
      if ( gfl_conn[i].IN_fl_[j] != -1 ) {
	val = gfl_conn[gfl_conn[i].IN_fl_[j]].level[time-1] + gfl_conn[i].IN_c[j];
      } else {
	val = gfl_conn[i].IN_c[j];
      }
      if ( val <= 0 ) {
	/* that one does not help us at all.
	 */
	continue;
      }
      /* if ef is already selected here, we can not use it anymore;
       * else, record it as selected.
       */
      if ( gef_conn[ef].in_plan == time - 1 ) {
	continue;
      } else {
	if ( gef_conn[ef].in_plan == INFINITY ) {
	  lin_plan_E[lnum_in_plan_E++] = ef;
	}
	gef_conn[ef].in_plan = time - 1;
      }
      /* do the usual stuff...
       */
      select_op( time, gef_conn[ef].op );
      introduce_benefits_and_enforcements( time, ef );
      introduce_pc_goals( time, ef );
      /* stop as soon as 
       * goal can be fulfilled one step below.
       */
      if ( number_comparison_holds( lf_goals_comp_at[time][i],
				    gfl_conn[i].level[time-1],
				    lf_goals_c_at[time][i] ) ) {
	break;
      }
    }
    /* now propagate the revised goal downward, and say we are finished with
     * this one.
     */
    update_f_goal( i, time-1, lf_goals_comp_at[time][i], lf_goals_c_at[time][i] );
    lf_goals_comp_at[time][i] = IGUAL;

    /* debug...
     */
    if ( !number_comparison_holds( lf_goals_comp_at[time-1][i],
				   gfl_conn[i].level[time-1],
				   lf_goals_c_at[time-1][i] ) ) {
      printf("\n\nall increasers applied yet goal not fulfilled.\n\n");
      exit( 1 );
    }
  }/* fluents at level time */



  /* now achieve also the remaining logic goals here.
   */
  for ( i = 0; i < lnum_goals_at[time]; i++ ) {
    ft = lgoals_at[time][i];

    if ( gft_conn[ft].is_true == time ) {
      /* fact already added by prev now selected op
       */
      continue;
    }

    if ( gcmd_line.debug ) {
      printf("\ngoal at level %d: ", time);
      print_ft_name(ft);
    }

    if ( !gcost_rplans ) {
      /* even in the non-cost-minimizing-rplans setting, 
       * choose the actions by their cost!
       * ... can only make the plan better...
       */
      min_preRPGcost = -1;
      min_e = -1;
      for ( j = 0; j < gft_conn[ft].num_A; j++ ) {
	ef = gft_conn[ft].A[j];
	if ( gef_conn[ef].level != time - 1 ) continue; 
	preRPGcost = gef_conn[ef].RPGcost;
	if ( min_preRPGcost == -1 || preRPGcost < min_preRPGcost ) {
	  min_preRPGcost = preRPGcost;
	  min_e = ef;
	}
      }
    } else {
      min_preRPGcost = -1;
      min_preRPGlevel = -1;
      min_e = -1;
      for ( j = 0; j < gft_conn[ft].num_A; j++ ) {
	ef = gft_conn[ft].A[j];
	/* in the cost setting, the level of the first supporter
	 * may actually be less than level-1... ! 
	 * (may have several smaller cost levels in between)
	 */
	if ( gef_conn[ef].level == -1 || gef_conn[ef].level >= time ) continue; 
	preRPGcost = gop_conn[gef_conn[ef].op].cost;
	preRPGlevel = 0;
	for ( k = 0; k < gef_conn[ef].num_PC; k++ ) {
	  preRPGcost += gft_conn[gef_conn[ef].PC[k]].RPGcost;
	  /* not sure if this is important once 0-cost actions are
	   * derived preds... anyway: with 0-cost acts,
	   * a fact may be acheievd in ini state and still have same cost
	   * as one that must be achieved still... hence prefer lower
	   * facts (also lower actions, see below...)
	   */
	  preRPGlevel += gft_conn[gef_conn[ef].PC[k]].level;
	}

	if ( gcmd_line.debug ) {
	  printf("\npossible achiever: ");
	  print_op_name(gef_conn[ef].op);
	  printf(" RPGcost: %f, own op-cost: %f, own level: %d, RPGlevel: %d", 
		 preRPGcost, gop_conn[gef_conn[ef].op].cost, gef_conn[ef].level, preRPGlevel);
	}
	if ( min_preRPGcost == -1 || preRPGcost < min_preRPGcost ) {
	  if ( gcmd_line.debug ) {
	    printf("\nTake-first!");
	  }
	  min_preRPGcost = preRPGcost;
	  min_preRPGlevel = preRPGlevel;
	  min_ownlevel = gef_conn[ef].level;
	  min_e = ef;
	  continue;
	}
	if ( preRPGcost = min_preRPGcost && 
	     (preRPGlevel < min_preRPGlevel || min_ownlevel > gef_conn[ef].level) ) {
	  if ( gcmd_line.debug ) {
	    printf("\nTake-later!");
	  }
	  min_preRPGcost = preRPGcost;
	  min_preRPGlevel = preRPGlevel;
	  min_e = ef;
	}
      }
    }
    ef = min_e;

    if ( min_e == -1 ) {
      printf("\nDEBUG ME: no supporting effect found\n");
      exit( 1 );
    }

    /* if ef is already selected, we can not use it anymore;
     * else, record it as selected.
     *
     * actually it can't happen here that the ef
     * is already selected as then the goal is true already.
     * nevermind.
     */
    if ( gef_conn[ef].in_plan == time - 1 ) {
      continue;
    } else {
      if ( gef_conn[ef].in_plan == INFINITY ) {
	lin_plan_E[lnum_in_plan_E++] = ef;
      }
      gef_conn[ef].in_plan = time - 1;
    }
    select_op( time, gef_conn[ef].op );
    introduce_benefits_and_enforcements( time, ef );
    introduce_pc_goals( time, ef );
  }

}



void enforce_artificial_goal( int fl, int time )

{

  int i;

  if ( !gfl_conn[fl].artificial ) {
    printf("\n\ntrying to enforce non-artificial goal\n\n");
    exit( 1 );
  }

  /* for now, we simply require the maximum levels of all 
   * composites.
   */
  for ( i = 0; i < gfl_conn[fl].num_lnf; i++ ) {
    update_f_goal( gfl_conn[fl].lnf_F[i], time, GEQ,
		   gfl_conn[gfl_conn[fl].lnf_F[i]].level[time] );
  }
  
}



void select_op( int time, int op )

{

  if ( gop_conn[op].is_used != time - 1 ) {
    /* we don't have this op yet at this level
     * --> count it and record it for setting back info.
     */
    if ( gop_conn[op].is_used == INFINITY ) {
      lused_O[lnum_used_O++] = op;
    }
    gop_conn[op].is_used = time - 1;
    /* do NOT count axiom-ops into the heuristic value!
     */
    if ( !gop_conn[op].axiom ) {
      lh++;
      /* this is where the cost-to-goal heuristic is computed!
       *
       * NOTE that this is used for any cost-minimizing search,
       * regardless of whether or not we also use cost-minimizing 
       * rplans!!
       */
      if ( gcost_minimizing ) {
	gh_cost += gop_conn[op].cost;
      }
    }
    if ( gcmd_line.display_info == 127 ) {
      printf("\n                       ");
      print_op_name( op );
    }
  }

}



void introduce_benefits_and_enforcements( int time, int ef )

{

  int k, l, ft, fl;
  float val;

  for ( k = 0; k < gef_conn[ef].num_A; k++ ) {
    ft = gef_conn[ef].A[k];
    gft_conn[ft].is_true = time;
    if ( !gft_conn[ft].ch ) {
      lch_F[lnum_ch_F++] = ft;
      gft_conn[ft].ch = TRUE;
    }
  }

  for ( k = 0; k < gef_conn[ef].num_AS; k++ ) {
    fl = gef_conn[ef].AS_fl[k];
    if ( lf_goals_comp_at[time][fl] == IGUAL ) {
      continue;
    }
    if ( !assign_value( ef, time - 1, k, &val ) ) {
      continue;
    }
    if ( number_comparison_holds( lf_goals_comp_at[time][fl],
				  val,
				  lf_goals_c_at[time][fl] ) ) {
      /* this effect assigns what we wanted there. enforce its
       * dependency fluent fl_
       */
      lf_goals_comp_at[time][fl] = IGUAL;
      enforce_assign( ef, time - 1, k );
    }
  }
  for ( k = 0; k < gef_conn[ef].num_IN; k++ ) {
    fl = gef_conn[ef].IN_fl[k];
    if ( lf_goals_comp_at[time][fl] == IGUAL ) {
      continue;
    }
    if ( !increase_value( ef, time - 1, k, &val ) ) {
      continue;
    }
    if ( val > 0 ) {
      lf_goals_c_at[time][fl] -= val;
      enforce_increase( ef, time - 1, k );
    }
  }

  for ( k = 0; k < gef_conn[ef].num_I; k++ ) {
    if ( gef_conn[gef_conn[ef].I[k]].in_plan == time - 1 ) {
      continue;
    } else {
      if ( gef_conn[gef_conn[ef].I[k]].in_plan == INFINITY ) {
	lin_plan_E[lnum_in_plan_E++] = gef_conn[ef].I[k];
      }
      gef_conn[gef_conn[ef].I[k]].in_plan = time - 1;
    }
    for ( l = 0; l < gef_conn[gef_conn[ef].I[k]].num_A; l++ ) {
      ft = gef_conn[gef_conn[ef].I[k]].A[l];
      gft_conn[ft].is_true = time;
      if ( !gft_conn[ft].ch ) {
	lch_F[lnum_ch_F++] = ft;
	gft_conn[ft].ch = TRUE;
      }
    }
    for ( l = 0; l < gef_conn[gef_conn[ef].I[k]].num_AS; l++ ) {
      fl = gef_conn[gef_conn[ef].I[k]].AS_fl[l];
      if ( lf_goals_comp_at[time][fl] == IGUAL ) {
	continue;
      }
      if ( !assign_value( gef_conn[ef].I[k], time - 1, l, &val ) ) {
	continue;
      }
      if ( number_comparison_holds( lf_goals_comp_at[time][fl],
				    val,
				    lf_goals_c_at[time][fl] ) ) {
	lf_goals_comp_at[time][fl] = IGUAL;
	enforce_assign( gef_conn[ef].I[k], time - 1, l );
      }
    }
    for ( l = 0; l < gef_conn[gef_conn[ef].I[k]].num_IN; l++ ) {
      fl = gef_conn[gef_conn[ef].I[k]].IN_fl[l];
      if ( lf_goals_comp_at[time][fl] == IGUAL ) {
	continue;
      }
      if ( !increase_value( gef_conn[ef].I[k], time - 1, l, &val ) ) {
	continue;
      }
      if ( val > 0 ) {
	lf_goals_c_at[time][fl] -= val;
	enforce_increase( gef_conn[ef].I[k], time - 1, l );
      }
    }
  }/* implied effects */

}



Bool assign_value( int ef, int at_time, int nr, float *val )

{

  if ( gef_conn[ef].AS_fl_[nr] == -1 ) {
    /* no dependency.
     */
    *val = gef_conn[ef].AS_c[nr];
    return TRUE;
  }

  if ( !gfl_conn[gef_conn[ef].AS_fl_[nr]].def[at_time] ) {
    return FALSE;
  }
  
  *val = gfl_conn[gef_conn[ef].AS_fl_[nr]].level[at_time] + gef_conn[ef].AS_c[nr];
  return TRUE;

}



Bool increase_value( int ef, int at_time, int nr, float *val )

{

  if ( gef_conn[ef].IN_fl_[nr] == -1 ) {
    /* no dependency.
     */
    *val = gef_conn[ef].IN_c[nr];
    return TRUE;
  }

  if ( !gfl_conn[gef_conn[ef].IN_fl_[nr]].def[at_time] ) {
    return FALSE;
  }
  
  *val = gfl_conn[gef_conn[ef].IN_fl_[nr]].level[at_time] + gef_conn[ef].IN_c[nr];
  return TRUE;

}



void enforce_assign( int ef, int at_time, int nr )

{

  if ( gef_conn[ef].AS_fl_[nr] == -1 ) {
    return;
  }

  if ( !gfl_conn[gef_conn[ef].AS_fl_[nr]].def[at_time] ) {
    printf("\n\ntrying to enforce an undefined value.\n\n");
    exit( 1 );
  }

  /* for now, simply require the maximum benefit of the effect.
   */
  update_f_goal( gef_conn[ef].AS_fl_[nr], at_time, GEQ,
		 gfl_conn[gef_conn[ef].AS_fl_[nr]].level[at_time] );

}



void enforce_increase( int ef, int at_time, int nr )

{

  if ( gef_conn[ef].IN_fl_[nr] == -1 ) {
    return;
  }

  if ( !gfl_conn[gef_conn[ef].IN_fl_[nr]].def[at_time] ) {
    printf("\n\ntrying to enforce an undefined value.\n\n");
    exit( 1 );
  }

  /* for now, simply require the maximum benefit of the effect.
   */
  update_f_goal( gef_conn[ef].IN_fl_[nr], at_time, GEQ,
		 gfl_conn[gef_conn[ef].IN_fl_[nr]].level[at_time] );

}



void introduce_pc_goals( int time, int ef )

{

  int k, l, ft, fl;
  float val;
  Comparator comp;

  /* now introduce the new goals
   */
  for ( k = 0; k < gef_conn[ef].num_PC; k++ ) {
    ft = gef_conn[ef].PC[k];
    if ( gft_conn[ft].is_goal ) {
      /* this fact already is a goal
       */
      continue;
    }
    lgoals_at[gft_conn[ft].level][lnum_goals_at[gft_conn[ft].level]++] = ft;
    gft_conn[ft].is_goal = TRUE;
    if ( !gft_conn[ft].ch ) {
      lch_F[lnum_ch_F++] = ft;
      gft_conn[ft].ch = TRUE;
    }
  }
  /* now for the numeric precs.
   */
  for ( k = 0; k < gef_conn[ef].num_f_PC; k++ ) {
    fl = gef_conn[ef].f_PC_fl[k];
    val = gef_conn[ef].f_PC_c[k];
    comp = gef_conn[ef].f_PC_comp[k];
    /* determine the first level where prec can be fulfilled.
     */
    for ( l = 0; l < time; l++ ) {
      if ( !gfl_conn[fl].def[l] ) continue;
      if ( number_comparison_holds( comp, gfl_conn[fl].level[l], val ) ) break;
    }
    if ( l >= time ) {
      printf("\n\nnumeric prec not reached in 1P??\n\n");
      exit( 1 );
    }
    /* if new requirement is stronger than old, then insert it.
     */
    update_f_goal( fl, l, comp, val );
  }

}



void update_f_goal( int fl, int time, Comparator comp, float val )

{

  if ( lf_goals_comp_at[time][fl] == IGUAL ) {
    lf_goals_comp_at[time][fl] = comp;
    lf_goals_c_at[time][fl] = val;
    return;
  }

  if ( lf_goals_c_at[time][fl] < val ) {
    lf_goals_comp_at[time][fl] = comp;
    lf_goals_c_at[time][fl] = val;
    return;
  }

  if ( lf_goals_c_at[time][fl] == val &&
       comp == GE ) {
    lf_goals_comp_at[time][fl] = GE;
  }
   
}



void reset_search_info( void )

{

  int i;

  for ( i = 0; i < lnum_ch_F; i++ ) {
    gft_conn[lch_F[i]].is_true = INFINITY;
    gft_conn[lch_F[i]].is_goal = FALSE;
    gft_conn[lch_F[i]].ch = FALSE;
  }
  lnum_ch_F = 0;

  for ( i = 0; i < lnum_used_O; i++ ) {
    gop_conn[lused_O[i]].is_used = INFINITY;
  }
  lnum_used_O = 0;
  
  for ( i = 0; i < lnum_in_plan_E; i++ ) {
    gef_conn[lin_plan_E[i]].in_plan = INFINITY;
  }
  lnum_in_plan_E = 0;

}



void collect_H_info( void )

{

  static Bool first_call = TRUE;

  int i, j, ft, ef, op;
  float val;

  if ( first_call ) {
    gH = ( int * ) calloc( gnum_op_conn, sizeof( int ) );
    gnum_H = 0;
    first_call = FALSE;
  }

  for ( i = 0; i < gnum_H; i++ ) {
    gop_conn[gH[i]].is_in_H = FALSE;
  }

  /* first the logical guys
   */
  gnum_H = 0;
  for ( i = lnum_goals_at[1] - 1; i >= 0; i-- ) {
    ft = lgoals_at[1][i];

    for ( j = 0; j < gft_conn[ft].num_A; j++ ) {
      ef = gft_conn[ft].A[j];
      if ( gef_conn[ef].level != 0 ) {
	continue;
      }
      op = gef_conn[ef].op;

      if ( gop_conn[op].is_in_H ) {
	continue;
      }
      if ( gop_conn[op].axiom ) {
	continue;
      }

      gop_conn[op].is_in_H = TRUE;
      gH[gnum_H++] = op;
    }
  }

  /* then the numerical ones.
   */
  for ( i = 0; i < gnum_real_fl_conn; i++ ) {
    if ( lHcomp[i] == IGUAL ) {
      /* don't need this one at all.
       */
      continue;
    }
    if ( gfl_conn[i].def[0] &&
	 number_comparison_holds( lHcomp[i], gfl_conn[i].level[0], lHc[i] ) ) {
      /* this one's already ok initially... was only pushed down
       * through RPG.
       */
      continue;
    }
				  
    /* assigners
     */
    for ( j = 0; j < gfl_conn[i].num_AS; j++ ) {
      ef = gfl_conn[i].AS[j];
      op = gef_conn[ef].op;
      if ( gop_conn[op].is_in_H ) {
	continue;
      }

      if ( gef_conn[ef].level != 0 ) {
	continue;
      }
      if ( gef_conn[ef].illegal ) {
	continue;
      }
      if ( gfl_conn[i].AS_fl_[j] != -1 &&
	   !gfl_conn[gfl_conn[i].AS_fl_[j]].def[0] ) {
	continue;
      }
      if ( gfl_conn[i].AS_fl_[j] == -1 ) {
	val = gfl_conn[i].AS_c[j];
      } else {
	val = gfl_conn[gfl_conn[i].AS_fl_[j]].level[0] + gfl_conn[i].AS_c[j];
      }
      if ( !number_comparison_holds( lHcomp[i], val, lHc[i] ) ) {
	continue;
      }

      gop_conn[op].is_in_H = TRUE;
      gH[gnum_H++] = op;
    }

    if ( !gfl_conn[i].def[0] ) {
      /* gotta be assigned.
       */
      continue;
    }

    /* increasers
     */
    for ( j = 0; j < gfl_conn[i].num_IN; j++ ) {
      ef = gfl_conn[i].IN[j];
      op = gef_conn[ef].op;
      if ( gop_conn[op].is_in_H ) {
	continue;
      }

      if ( gef_conn[ef].level != 0 ) {
	continue;
      }
      if ( gef_conn[ef].illegal ) {
	continue;
      }
      if ( gfl_conn[i].IN_fl_[j] != -1 &&
	   !gfl_conn[gfl_conn[i].IN_fl_[j]].def[0] ) {
	continue;
      }
      if ( gfl_conn[i].IN_fl_[j] == -1 ) {
	val = gfl_conn[i].IN_c[j];
      } else {
	val = gfl_conn[gfl_conn[i].IN_fl_[j]].level[0] + gfl_conn[i].IN_c[j];
      }
      if ( val <= 0 ) {
	continue;
      }

      gop_conn[op].is_in_H = TRUE;
      gH[gnum_H++] = op;
    }
  }

  if ( gcmd_line.display_info == 128 ) {
    printf("\ncollected H: ");
    for ( i = 0; i < gnum_H; i++ ) {
      print_op_name( gH[i] );
      printf("\n              ");
    }
  }

}
