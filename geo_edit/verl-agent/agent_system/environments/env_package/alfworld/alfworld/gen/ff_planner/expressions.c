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



/***********************************************************************
 * File: expressions.c
 * Description: functions for handling numerical expressions
 *
 *              - general utilities:
 *                comparisons between numbers etc.
 *
 *              - LNF compilation:
 *                normalization of expressions
 *                translation of subtractions
 *
 *              - LNF post-processing:
 *                summarization of effects
 *                encoding of non-minimal LNFs
 *
 * Author: Joerg Hoffmann 2001
 *
 *********************************************************************/ 








#include "ff.h"

#include "output.h"
#include "memory.h"

#include "expressions.h"















/*******************************************************
 * SIMPLE UTILITIES
 *******************************************************/
















Bool number_comparison_holds( Comparator c, float l, float r )

{

  switch ( c ) {
  case LE:
    if ( l < r ) return TRUE;
    break;
  case LEQ:
    if ( l <= r ) return TRUE;
    break;
  case EQ:
    if ( l == r ) return TRUE;
    break;
  case GEQ:
    if ( l >= r ) return TRUE;
    break;
  case GE:
    if ( l > r ) return TRUE;
    break;
  case IGUAL:
    /* technical for non-required fluents
     */
    return TRUE;
  default:
    printf("\n\nillegal comparator %d in number comp holds", c);
    exit( 1 );
  }

  return FALSE;

}





















/*******************************************************
 * MACHINERY FOR LNF TRANSFORMATION!!!!!!
 *******************************************************/
























Bool transform_to_LNF( void )

{

  if ( !is_linear_task() ) {
    return FALSE;
  }

  normalize_expressions();
  if ( gcmd_line.display_info == 121 ) {
    printf("\n\nnormalized expressions representation is:\n\n");
    print_lnf_representation();
  }

  translate_subtractions();
  if ( gcmd_line.display_info == 122 ) {
    printf("\n\nLNF : translated subtractions representation is:\n\n");
    print_lnf_representation();
  }

  /* LNF computed. start post-processing.
   */

  /* do same-cond effects etc. summarization here so as to have
   * as tight as possible an encoded LNF representation.
   */
  summarize_effects();
  if ( gcmd_line.display_info == 123 ) {
    printf("\n\nLNF - summarized effects representation is:\n\n");
    print_lnf_representation();
  }

  encode_lfns_as_artificial_fluents();
  /* optimization is translated into minimizing
   * effect costs... here, determine the cost that
   * each effect has.
   *
   * returns TRUE if a non-trivial optimization expression
   * could be established.
   */
  if ( setup_effect_costs() ) {
    if ( gcmd_line.display_info > 1 ) {
      printf("\nmetric established (normalized to minimize): ");
      print_LnfExpNode( &glnf_metric );
    }
    goptimization_established = TRUE;
  }
  if ( gcmd_line.display_info == 124 ) {
    printf("\n\nencoded LNF representation is:\n\n");
    print_lnf_representation();
  }

  return TRUE;

}



/* simple syntax check
 */
Bool is_linear_task( void )

{

  Action *a;
  ActionEffect *e;
  int i, j;

  for ( a = gactions; a; a = a->next ) {
    /* preconds
     */
    for ( i = 0; i < a->num_numeric_preconds; i++ ) {
      if ( !is_linear_expression( a->numeric_preconds_lh[i] ) ) {
	return FALSE;
      }
      if ( !is_linear_expression( a->numeric_preconds_rh[i] ) ) {
	return FALSE;
      }
    }

    /* effects
     */
    for ( i = 0; i < a->num_effects; i++ ) {
      e = &(a->effects[i]);
      for ( j = 0; j < e->num_numeric_conditions; j++ ) {
	if ( !is_linear_expression( e->numeric_conditions_lh[j] ) ) {
	  return FALSE;
	}
	if ( !is_linear_expression( e->numeric_conditions_rh[j] ) ) {
	  return FALSE;
	}
      }

      if ( e->illegal ) {
	/* we don't care whether that one's ok or not-
	 * it won't be applied anyway.
	 */
	continue;
      }

      for ( j = 0; j < e->num_numeric_effects; j++ ) {
	if ( e->numeric_effects_neft[j] != INCREASE &&
	     e->numeric_effects_neft[j] != DECREASE &&
	     e->numeric_effects_neft[j] != ASSIGN ) {
	  return FALSE;
	}
	if ( !is_linear_expression( e->numeric_effects_rh[j] ) ) {
	  return FALSE;
	}
      }
    }
  }

  /* goal condition also...
   */
  for ( i = 0; i < gnum_numeric_goal; i++ ) {
    if ( !is_linear_expression( gnumeric_goal_lh[i] ) ) {
      return FALSE;
    }
    if ( !is_linear_expression( gnumeric_goal_rh[i] ) ) {
      return FALSE;
    }
  }

  if ( gmetric != NULL ) {
    if ( !is_linear_expression( gmetric ) ) {
      if ( gcmd_line.display_info ) {
	printf("\nwarning: metric is no linear expression. defaulting to plan length.");
      }
      free_ExpNode( gmetric );
      gmetric = NULL;      
    }
  }

  return TRUE;

}



Bool is_linear_expression( ExpNode *n )

{

  switch ( n->connective ) {
  case MU:
    if ( !is_linear_expression( n->leftson ) ||
	 !is_linear_expression( n->rightson ) ) {
      return FALSE;
    }
    if ( n->leftson->connective != NUMBER &&
	 n->rightson->connective != NUMBER ) {
      return FALSE;
    }
    break;
  case DI:
    if ( !is_linear_expression( n->leftson ) ||
	 n->rightson->connective != NUMBER ) {
      return FALSE;
    }
    break;
  case AD:
  case SU:
    if ( !is_linear_expression( n->leftson ) ||
	 !is_linear_expression( n->rightson ) ) {
      return FALSE;
    }
    break;
  case MINUS:
    if ( !is_linear_expression( n->son ) ) {
      return FALSE;
    }
    break;
  case NUMBER:
  case FHEAD:
    break;
  default:
    printf("\n\nis linear exp: wrong specifier %d",
	   n->connective);
    exit( 1 );
  }

  return TRUE;

}



void print_lnf_representation( void )

{

  int i;
  Action *a;

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
      print_lnf_Action( a );
    }
  }
  printf("\n\n--------------------GOAL REACHED ops-----------\n\n");
  for ( a = gactions; a; a = a->next ) {
    if ( !a->norm_operator &&
	 !a->pseudo_action ) {
      print_lnf_Action( a );
    }
  }
  
  printf("\n\ninitial state is:\n\n");
  print_State( ginitial_state );
  
  printf("\n\ngoal is:\n\n");
  for ( i = 0; i < gnum_logic_goal; i++ ) {
    print_ft_name( glogic_goal[i] );
    printf("\n");
  }
  for ( i = 0; i < gnum_lnf_goal; i++ ) {
    switch ( glnf_goal_comp[i] ) {
    case GEQ:
      printf("(>= ");
      break;
    case GE:
      printf("(> ");
      break;
    default:
      printf("\nwrong comparator in lnf goal %d\n\n", glnf_goal_comp[i]);
      exit( 1 );
    }
    print_LnfExpNode( glnf_goal_lh[i] );
    printf(" %f", glnf_goal_rh[i]);
    printf(")\n");
  }

  if ( gmetric ) {
    printf("\n\nmetric is (minimize) (constant part skipped):\n");
    print_LnfExpNode( &glnf_metric );
  } else {
    printf("\n\nmetric: none, i.e. plan length\n");
  }

}


















/*******************************************************
 * SUBPART I: NORMALIZE THE EXPRESSIONS
 *******************************************************/

















/* local globals.
 */

Comparator lcomp;

int lF[MAX_LNF_F];
float lC[MAX_LNF_F];
int lnum_F;

float lc;











void normalize_expressions( void )

{

  Action *a, *p, *t;
  ActionEffect *e;
  int i, j, k;
  Bool eq;
  LnfExpNode *lnf;

  /* first, pre-normalize all the expressions, i.e. translate
   * divisions, and push muliplications downwards.
   */
  for ( i = 0; i < gnum_numeric_goal; i++ ) {
    if ( !translate_divisions( &(gnumeric_goal_lh[i]) ) ) {
      printf("\n\nff: division by zero in goal. no plan will solve it.\n\n");
      exit( 1 );
    }
    push_multiplications_down( &(gnumeric_goal_lh[i]) );
    if ( !translate_divisions( &(gnumeric_goal_rh[i]) ) ) {
      printf("\n\nff: division by zero in goal. no plan will solve it.\n\n");
      exit( 1 );
    }
    push_multiplications_down( &(gnumeric_goal_rh[i]) );
  }

  a = gactions; p = NULL;
  while ( a ) {
    for ( i = 0; i < a->num_numeric_preconds; i++ ) {
      if ( !translate_divisions( &(a->numeric_preconds_lh[i]) ) ) break;
      push_multiplications_down( &(a->numeric_preconds_lh[i]) );
      if ( !translate_divisions( &(a->numeric_preconds_rh[i]) ) ) break;
      push_multiplications_down( &(a->numeric_preconds_rh[i]) );
    }
    if ( i < a->num_numeric_preconds ) {
      if ( gcmd_line.display_info ) {
	printf("\nwarning: division by zero in precond of ");
	print_Action_name( a );
	printf(". skipping action.");
      }
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

    for ( i = 0; i < a->num_effects; i++ ) {
      e = &(a->effects[i]);

      for ( j = 0; j < e->num_numeric_conditions; j++ ) {
	if ( !translate_divisions( &(e->numeric_conditions_lh[j]) ) ) break;
	push_multiplications_down( &(e->numeric_conditions_lh[j]) );
	if ( !translate_divisions( &(e->numeric_conditions_rh[j]) ) ) break;
	push_multiplications_down( &(e->numeric_conditions_rh[j]) );
      }
      if ( j < e->num_numeric_conditions ) break;

      if ( e->illegal ) {
	continue;
      }

      for ( j = 0; j < e->num_numeric_effects; j++ ) {
	if ( !translate_divisions( &(e->numeric_effects_rh[j]) ) ) break;
	push_multiplications_down( &(e->numeric_effects_rh[j]) );
      }
      if ( j < e->num_numeric_effects ) {
	if ( gcmd_line.display_info ) {
	  printf("\nwarning: division by zero in effect rh of ");
	  print_Action_name( a );
	  printf(". marking effect as illegal.");
	}
	e->illegal = TRUE;
      }
    }
    if ( i < a->num_effects ) {
      if ( gcmd_line.display_info ) {
	printf("\nwarning: division by zero in effect cond of ");
	print_Action_name( a );
	printf(". skipping action.");
      }
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
    
    p = a;
    a = a->next;
  }
  if ( gmetric != NULL ) {
    if ( !translate_divisions( &gmetric ) ) {
      if ( gcmd_line.display_info ) {
	printf("\nwarning: division by zero in metric. replaced with plan length.");
      }
      free_ExpNode( gmetric );
      gmetric = NULL;
    }
    push_multiplications_down( &gmetric );
  }

  /* now, collect the normalized representations of all expressions.
   */
  for ( a = gactions; a; a = a->next ) {
    /* preconds
     */
    a->lnf_preconds_comp = ( Comparator * ) calloc( MAX_LNF_COMPS, sizeof( Comparator ) );
    a->lnf_preconds_lh = ( LnfExpNode_pointer * ) calloc( MAX_LNF_COMPS, sizeof( LnfExpNode_pointer ) );
    a->lnf_preconds_rh = ( float * ) calloc( MAX_LNF_COMPS, sizeof( float ) );
    a->num_lnf_preconds = 0;
    for ( i = 0; i < a->num_numeric_preconds; i++ ) {
      if ( a->num_lnf_preconds == MAX_LNF_COMPS ) {
	printf("\n\nincrease MAX_LNF_COMPS! currently %d\n\n", MAX_LNF_COMPS);
	exit( 1 );
      }
      eq = FALSE;
      if ( a->numeric_preconds_comp[i] == EQ ) {
	eq = TRUE;
	a->numeric_preconds_comp[i] = LEQ;
      }
      put_comp_into_normalized_locals( a->numeric_preconds_comp[i],
				       a->numeric_preconds_lh[i],
				       a->numeric_preconds_rh[i] );
      a->lnf_preconds_comp[a->num_lnf_preconds] = lcomp;
      a->lnf_preconds_lh[a->num_lnf_preconds] = new_LnfExpNode();
      lnf = a->lnf_preconds_lh[a->num_lnf_preconds];
      for ( j = 0; j < lnum_F; j++ ) {
	if ( lC[j] == 0 ) continue;
	if ( lC[j] > 0 ) {
	  lnf->pF[lnf->num_pF] = lF[j];
	  lnf->pC[lnf->num_pF++] = lC[j];
	} else {
	  lnf->nF[lnf->num_nF] = lF[j];
	  lnf->nC[lnf->num_nF++] = (-1) * lC[j];
	}
      }
      a->lnf_preconds_rh[a->num_lnf_preconds] = lc;
      a->num_lnf_preconds++;
      if ( eq ) {
	if ( a->num_lnf_preconds == MAX_LNF_COMPS ) {
	  printf("\n\nincrease MAX_LNF_COMPS! currently %d\n\n", MAX_LNF_COMPS);
	  exit( 1 );
	}
	a->numeric_preconds_comp[i] = EQ;
	put_comp_into_normalized_locals( GEQ,
					 a->numeric_preconds_lh[i],
					 a->numeric_preconds_rh[i] );
	a->lnf_preconds_comp[a->num_lnf_preconds] = lcomp;
	a->lnf_preconds_lh[a->num_lnf_preconds] = new_LnfExpNode();
	lnf = a->lnf_preconds_lh[a->num_lnf_preconds];
	for ( j = 0; j < lnum_F; j++ ) {
	  if ( lC[j] == 0 ) continue;
	  if ( lC[j] > 0 ) {
	    lnf->pF[lnf->num_pF] = lF[j];
	    lnf->pC[lnf->num_pF++] = lC[j];
	  } else {
	    lnf->nF[lnf->num_nF] = lF[j];
	    lnf->nC[lnf->num_nF++] = (-1) * lC[j];
	  }
	}
	a->lnf_preconds_rh[a->num_lnf_preconds] = lc;
	a->num_lnf_preconds++;
      }
    }

    /* effects
     */
    for ( i = 0; i < a->num_effects; i++ ) {
      e = &(a->effects[i]);

      e->lnf_conditions_comp = ( Comparator * ) calloc( MAX_LNF_COMPS, sizeof( Comparator ) );
      e->lnf_conditions_lh = ( LnfExpNode_pointer * ) calloc( MAX_LNF_COMPS, sizeof( LnfExpNode_pointer ) );
      e->lnf_conditions_rh = ( float * ) calloc( MAX_LNF_COMPS, sizeof( float ) );
      e->num_lnf_conditions = 0;
      for ( j = 0; j < e->num_numeric_conditions; j++ ) {
	if ( e->num_lnf_conditions == MAX_LNF_COMPS ) {
	  printf("\n\nincrease MAX_LNF_COMPS! currently %d\n\n", MAX_LNF_COMPS);
	  exit( 1 );
	}
	eq = FALSE;
	if ( e->numeric_conditions_comp[j] == EQ ) {
	  eq = TRUE;
	  e->numeric_conditions_comp[j] = LEQ;
	}
	put_comp_into_normalized_locals( e->numeric_conditions_comp[j],
					 e->numeric_conditions_lh[j],
					 e->numeric_conditions_rh[j] );
	e->lnf_conditions_comp[e->num_lnf_conditions] = lcomp;
	e->lnf_conditions_lh[e->num_lnf_conditions] = new_LnfExpNode();
	lnf = e->lnf_conditions_lh[e->num_lnf_conditions];
	for ( k = 0; k < lnum_F; k++ ) {
	  if ( lC[k] == 0 ) continue;
	  if ( lC[k] > 0 ) {
	    lnf->pF[lnf->num_pF] = lF[k];
	    lnf->pC[lnf->num_pF++] = lC[k];
	  } else {
	    lnf->nF[lnf->num_nF] = lF[k];
	    lnf->nC[lnf->num_nF++] = (-1) * lC[k];
	  }
	}
	e->lnf_conditions_rh[e->num_lnf_conditions] = lc;
	e->num_lnf_conditions++;
	if ( eq ) {
	  if ( e->num_lnf_conditions == MAX_LNF_COMPS ) {
	    printf("\n\nincrease MAX_LNF_COMPS! currently %d\n\n", MAX_LNF_COMPS);
	    exit( 1 );
	  }
	  e->numeric_conditions_comp[j] = EQ;
	  put_comp_into_normalized_locals( GEQ,
					   e->numeric_conditions_lh[j],
					   e->numeric_conditions_rh[j] );
	  e->lnf_conditions_comp[e->num_lnf_conditions] = lcomp;
	  e->lnf_conditions_lh[e->num_lnf_conditions] = new_LnfExpNode();
	  lnf = e->lnf_conditions_lh[e->num_lnf_conditions];
	  for ( k = 0; k < lnum_F; k++ ) {
	    if ( lC[k] == 0 ) continue;
	    if ( lC[k] > 0 ) {
	      lnf->pF[lnf->num_pF] = lF[k];
	      lnf->pC[lnf->num_pF++] = lC[k];
	    } else {
	      lnf->nF[lnf->num_nF] = lF[k];
	      lnf->nC[lnf->num_nF++] = (-1) * lC[k];
	    }
	  }
	  e->lnf_conditions_rh[e->num_lnf_conditions] = lc;
	  e->num_lnf_conditions++;
	}
      }

      if ( e->illegal ) {
	/* we do have the LNF to know whether the effect appears.
	 * if it does, then this one is illegal anyway, remembered
	 * in inst final due to undefined fl access.
	 *
	 * if it is LEGAL, then all fluents we're gonna find and
	 * collect below are relevant!!!
	 */
	continue;
      }
      
      e->lnf_effects_neft = ( NumericEffectType * ) calloc( MAX_LNF_EFFS, sizeof( NumericEffectType ) );
      e->lnf_effects_fl = ( int * ) calloc( MAX_LNF_EFFS, sizeof( int ) );
      e->lnf_effects_rh = ( LnfExpNode_pointer * ) calloc( MAX_LNF_EFFS, sizeof( LnfExpNode_pointer ) );
      e->num_lnf_effects = 0;
      for ( j = 0; j < e->num_numeric_effects; j++ ) {
	if ( e->num_lnf_effects == MAX_LNF_EFFS ) {
	  printf("\n\nincrease MAX_LNF_EFFS! currently %d\n\n", MAX_LNF_EFFS);
	  exit( 1 );
	}
	e->lnf_effects_neft[e->num_lnf_effects] = e->numeric_effects_neft[j];
	e->lnf_effects_fl[e->num_lnf_effects] = e->numeric_effects_fl[j];
	lnum_F = 0;
	lc = 0;
	if ( e->lnf_effects_neft[e->num_lnf_effects] == DECREASE ) {
	  collect_normalized_locals( e->numeric_effects_rh[j], FALSE );
	  e->lnf_effects_neft[e->num_lnf_effects] = INCREASE;
	} else {
	  collect_normalized_locals( e->numeric_effects_rh[j], TRUE );
	}
	e->lnf_effects_rh[e->num_lnf_effects] = new_LnfExpNode();
	lnf = e->lnf_effects_rh[e->num_lnf_effects];
	for ( k = 0; k < lnum_F; k++ ) {
	  if ( lC[k] == 0 ) continue;
	  if ( lC[k] > 0 ) {
	    lnf->pF[lnf->num_pF] = lF[k];
	    lnf->pC[lnf->num_pF++] = lC[k];
	  } else {
	    lnf->nF[lnf->num_nF] = lF[k];
	    lnf->nC[lnf->num_nF++] = (-1) * lC[k];
	  }
	}
	e->lnf_effects_rh[e->num_lnf_effects]->c = lc;
	e->num_lnf_effects++;
      }
    }
  }

  /* goal condition also...
   */
  glnf_goal_comp = ( Comparator * ) calloc( MAX_LNF_COMPS, sizeof( Comparator ) );
  glnf_goal_lh = ( LnfExpNode_pointer * ) calloc( MAX_LNF_COMPS, sizeof( LnfExpNode_pointer ) );
  glnf_goal_rh = ( float * ) calloc( MAX_LNF_COMPS, sizeof( float ) );
  gnum_lnf_goal = 0;
  for ( i = 0; i < gnum_numeric_goal; i++ ) {
    if ( gnum_lnf_goal == MAX_LNF_COMPS ) {
      printf("\n\nincrease MAX_LNF_COMPS! currently %d\n\n", MAX_LNF_COMPS);
      exit( 1 );
    }
    eq = FALSE;
    if ( gnumeric_goal_comp[i] == EQ ) {
      eq = TRUE;
      gnumeric_goal_comp[i] = LEQ;
    }
    put_comp_into_normalized_locals( gnumeric_goal_comp[i],
				     gnumeric_goal_lh[i],
				     gnumeric_goal_rh[i] );
    glnf_goal_comp[gnum_lnf_goal] = lcomp;
    glnf_goal_lh[gnum_lnf_goal] = new_LnfExpNode();
    lnf = glnf_goal_lh[gnum_lnf_goal];
    for ( j = 0; j < lnum_F; j++ ) {
      if ( lC[j] == 0 ) continue;
      if ( lC[j] > 0 ) {
	lnf->pF[lnf->num_pF] = lF[j];
	lnf->pC[lnf->num_pF++] = lC[j];
      } else {
	lnf->nF[lnf->num_nF] = lF[j];
	lnf->nC[lnf->num_nF++] = (-1) * lC[j];
      }
    }
    glnf_goal_rh[gnum_lnf_goal] = lc;
    gnum_lnf_goal++;
    if ( eq ) {
      if ( gnum_lnf_goal == MAX_LNF_COMPS ) {
	printf("\n\nincrease MAX_LNF_COMPS! currently %d\n\n", MAX_LNF_COMPS);
	exit( 1 );
      }
      gnumeric_goal_comp[i] = EQ;
      put_comp_into_normalized_locals( GEQ,
				       gnumeric_goal_lh[i],
				       gnumeric_goal_rh[i] );
      glnf_goal_comp[gnum_lnf_goal] = lcomp;
      glnf_goal_lh[gnum_lnf_goal] = new_LnfExpNode();
      lnf = glnf_goal_lh[gnum_lnf_goal];
      for ( j = 0; j < lnum_F; j++ ) {
	if ( lC[j] == 0 ) continue;
	if ( lC[j] > 0 ) {
	  lnf->pF[lnf->num_pF] = lF[j];
	  lnf->pC[lnf->num_pF++] = lC[j];
	} else {
	  lnf->nF[lnf->num_nF] = lF[j];
	  lnf->nC[lnf->num_nF++] = (-1) * lC[j];
	}
      }
      glnf_goal_rh[gnum_lnf_goal] = lc;
      gnum_lnf_goal++;
    }
  }
  /* metric...
   */
  lnum_F = 0;
  lc = 0;
  glnf_metric.num_pF = 0;
  glnf_metric.num_nF = 0;
  glnf_metric.c = 0;
  collect_normalized_locals( gmetric, TRUE );
  lnf = &glnf_metric;
  for ( j = 0; j < lnum_F; j++ ) {
    if ( lC[j] == 0 ) continue;
    if ( lC[j] > 0 ) {
      lnf->pF[lnf->num_pF] = lF[j];
      lnf->pC[lnf->num_pF++] = lC[j];
    } else {
      lnf->nF[lnf->num_nF] = lF[j];
      lnf->nC[lnf->num_nF++] = (-1) * lC[j];
    }
  }


}



Bool translate_divisions( ExpNode **n )

{

  ExpNode *tmp;

  /* "dirty": also normalize multiplications so that the constant
   * is always on the left hand side ---
   * simplifies function below a lot.
   */
  switch ( (*n)->connective ) {
  case DI:
    /* rightson is number due to syntax check.
     */
    if ( (*n)->rightson->value == 0 ) {
      /* what needs to be done we can only decide further up.
       */
      printf("\nwarning: division by zero.");
      return FALSE;
    }
    if ( !translate_divisions( &((*n)->leftson) ) ) return FALSE;
    (*n)->connective = MU;
    (*n)->rightson->value = 1 / (*n)->rightson->value;
    tmp = (*n)->rightson;
    (*n)->rightson = (*n)->leftson;
    (*n)->leftson = tmp;
    break;
  case MU:
    if ( !translate_divisions( &((*n)->leftson) ) ) return FALSE;
    if ( !translate_divisions( &((*n)->rightson) ) ) return FALSE;
    if ( (*n)->rightson->connective == NUMBER ) {
      tmp = (*n)->rightson;
      (*n)->rightson = (*n)->leftson;
      (*n)->leftson = tmp;      
    }
    break;
  case AD:
  case SU:
    if ( !translate_divisions( &((*n)->leftson) ) ) return FALSE;
    if ( !translate_divisions( &((*n)->rightson) ) ) return FALSE;
    break;
  case MINUS:
    if ( !translate_divisions( &((*n)->son) ) ) return FALSE;
    break;
  case NUMBER:
  case FHEAD:
    break;
  default:
    printf("\n\ntranslate divisions: wrong specifier %d",
	   (*n)->connective);
    exit( 1 );
  }

  return TRUE;

}



void push_multiplications_down( ExpNode **n )

{

  ExpNode *tmp1, *tmp2;

  switch ( (*n)->connective ) {
  case MU:
    /* due to syntax check, at least one of sons is number,
     * 
     * due to above, it's the left one.
     * NOTE that this invariant is kept true troughout the
     * modifications done here.
     */
    if ( (*n)->rightson->connective == NUMBER ) {
      (*n)->connective = NUMBER;
      (*n)->value = (*n)->leftson->value * (*n)->rightson->value;
      free_ExpNode( (*n)->leftson );
      free_ExpNode( (*n)->rightson );
      (*n)->leftson = NULL;
      (*n)->rightson = NULL;
      break;	
    }
    if ( (*n)->rightson->connective == FHEAD ) {
      (*n)->connective = FHEAD;
      (*n)->fl = (*n)->rightson->fl;
      (*n)->c = (*n)->leftson->value;
      free_ExpNode( (*n)->leftson );
      free_ExpNode( (*n)->rightson );
      (*n)->leftson = NULL;
      (*n)->rightson = NULL;
      break;
    }
    if ( (*n)->rightson->connective == MINUS ) {
      (*n)->connective = MINUS;
      (*n)->son = (*n)->rightson;
      (*n)->son->connective = MU;
      (*n)->son->leftson = (*n)->leftson;
      (*n)->son->rightson = (*n)->rightson->son;
      (*n)->rightson = NULL;
      (*n)->leftson = NULL;
      (*n)->son->son = NULL;
      push_multiplications_down( &((*n)->son) );
      break;
    }
    if ( (*n)->rightson->connective == MU ) {
      (*n)->leftson->value *= (*n)->rightson->leftson->value;
      tmp1 = (*n)->rightson->rightson;
      (*n)->rightson->rightson = NULL;
      free_ExpNode( (*n)->rightson );
      (*n)->rightson = tmp1;
      push_multiplications_down( n );
      break;
    }
    
    /* rigthson is either AD or SU
     */
    tmp1 = new_ExpNode( NUMBER );
    tmp2 = new_ExpNode( NUMBER );
    tmp1->value = (*n)->leftson->value;
    tmp2->value = (*n)->leftson->value;
    
    (*n)->connective = (*n)->rightson->connective;
    (*n)->leftson->connective = MU;
    (*n)->rightson->connective = MU;
    (*n)->leftson->leftson = tmp1;
    (*n)->leftson->rightson = (*n)->rightson->leftson;      
    (*n)->rightson->leftson = tmp2;

    push_multiplications_down( &((*n)->leftson) );
    push_multiplications_down( &((*n)->rightson) );
    break;
  case AD:
  case SU:
    push_multiplications_down( &((*n)->leftson) );
    push_multiplications_down( &((*n)->rightson) );
    break;
  case MINUS:
    push_multiplications_down( &((*n)->son) );
    break;
  case NUMBER:
  case FHEAD:
    break;
  default:
    printf("\n\ntranslate divisions: wrong specifier %d",
	   (*n)->connective);
    exit( 1 );
  }

}



void put_comp_into_normalized_locals( Comparator comp,
				      ExpNode *lh,
				      ExpNode *rh )

{

  ExpNode *tmp;

  tmp = new_ExpNode( SU );

  /* initialisation of normalized locals
   */
  lnum_F = 0;
  lc = 0;

  lcomp = comp;

  /* if comparison is LE or LEQ, then subtract
   * left hand side from right hand side to obtain
   * new left hand side.
   */
  if ( lcomp == LE ) {
    tmp->leftson = rh;
    tmp->rightson = lh;
    collect_normalized_locals( tmp, TRUE );
    lcomp = GE;
    /* "subtract" the constant to get it to the right hand
     * side.
     */
    lc *= (-1);
    free( tmp );
    return;
  }
  if ( lcomp == LEQ ) {
    tmp->leftson = rh;
    tmp->rightson = lh;
    collect_normalized_locals( tmp, TRUE );
    lcomp = GEQ;
    lc *= (-1);
    free( tmp );
    return;
  }

  /* otherwise, subtract right hand side from left hand side.
   */
  tmp->leftson = lh;
  tmp->rightson = rh;
  collect_normalized_locals( tmp, TRUE );
  lc *= (-1);
  free( tmp );

}



void collect_normalized_locals( ExpNode *n, Bool positive )

{

  Bool negative = positive ? FALSE : TRUE;
  int i;

  if ( !n ) return;

  switch ( n->connective ) {
  case AD:
    collect_normalized_locals( n->leftson, positive );
    collect_normalized_locals( n->rightson, positive );
    break;
  case SU:
    collect_normalized_locals( n->leftson, positive );
    collect_normalized_locals( n->rightson, negative );
    break;
  case MINUS:
    collect_normalized_locals( n->son, negative );
    break;
  case NUMBER:
    if ( positive ) {
      lc += n->value;
    } else {
      lc -= n->value;
    }
    break;
  case FHEAD:
    if ( n->fl < 0 && n->fl != -2 ) {
      printf("\n\ncollecting non-relevant fluent for LNF!!\n\n");
      exit( 1 );
    }
    for ( i = 0; i < lnum_F; i++ ) {
      if ( lF[i] == n->fl ) break;
    }
    if ( i < lnum_F ) {
      lC[i] += positive ? n->c : ((-1) * n->c);
    } else { 
      if ( lnum_F == MAX_LNF_F ) {
	printf("\n\nincrease MAX_LNF_F! currently %d\n\n", MAX_LNF_F);
	exit( 1 );
      }
      lF[lnum_F] = n->fl;
      lC[lnum_F] = positive ? n->c : ((-1) * n->c);
      lnum_F++;
    }
    break;
  default:
    printf("\n\ncollect_normalized_locals: wrong specifier %d",
	   n->connective);
    exit( 1 );
  }

}





















/*******************************************************
 * SUBPART II: TRANSLATE THE SUBTRACTIONS
 *******************************************************/















/* local globals.
 */

int lminus_fluent[MAX_RELEVANT_FLUENTS];












void translate_subtractions( void )

{

  int i, fl;

  /* minus_fluent[fl] gives the number of the fluent that
   * takes on the negative value to fl, or -1 if there is
   * no such fluent.
   */
  for ( i = 0; i < MAX_RELEVANT_FLUENTS; i++ ) {
    lminus_fluent[i] = -1;
  }

  while ( TRUE ) {
    /* ex fl \in nF for pre, cond, eff or goal?
     */
    if ( !ex_fl_in_nF_of_pre_cond_eff_goal( &fl ) ) {
      /* no --> we are finished.
       */
      break;
    }
    if ( fl < 0 ) {
      if ( fl != -2 ) {
	printf("\n\nnon-relevant fluent in non-illegal part!\n\n");
	exit( 1 );
      } else {
	printf("\n\nwarning: total-time occurs negatively in metric. no optimization done.\n\n");
	glnf_metric.num_pF = 0;
	glnf_metric.num_nF = 0;
	continue;
      }
    }
    /* set the new number and name, incrementing 
     * gnum_relevant_fluents, and setting
     * minus_fluent value for both directions.
     */
    introduce_minus_fluent( fl );
    /* replace all occurences in effects and conds and goals
     */
    replace_fl_in_nF_with_minus_fl( fl );
    /* set the initial value of the new fluent
     */
    set_minus_fl_initial( fl );
    /* adjust the effects accordingly
     */
    introduce_minus_fl_effects( fl );
  }

}



Bool ex_fl_in_nF_of_pre_cond_eff_goal( int *fl )

{

  Action *a;
  ActionEffect *e;
  int i, j;

  for ( i = 0; i < gnum_lnf_goal; i++ ) {
    if ( glnf_goal_lh[i]->num_nF > 0 ) {
      *fl = glnf_goal_lh[i]->nF[0];
      return TRUE;
    }
  }

  for ( a = gactions; a; a = a->next ) {
    for ( i = 0; i < a->num_lnf_preconds; i++ ) {
      if ( a->lnf_preconds_lh[i]->num_nF > 0 ) {
	*fl = a->lnf_preconds_lh[i]->nF[0];
	return TRUE;
      }
    }

    for ( i = 0; i < a->num_effects; i++ ) {
      e = &(a->effects[i]);

      for ( j = 0; j < e->num_lnf_conditions; j++ ) {
	if ( e->lnf_conditions_lh[j]->num_nF > 0 ) {
	  *fl = e->lnf_conditions_lh[j]->nF[0];
	  return TRUE;
	}
      }

      if ( e->illegal ) {
	/* we don't care if there's something in here that
	 * wants to be translated.
	 */
	continue;
      }

      for ( j = 0; j < e->num_lnf_effects; j++ ) {
	if ( e->lnf_effects_rh[j]->num_nF > 0 ) {
	  *fl = e->lnf_effects_rh[j]->nF[0];
	  return TRUE;
	}
      }
    }
  }

  /* no need to throw costs away, even if we're not explicitly asked to 
   * minimize them
   */
  if ( (1 || gcost_minimizing) && glnf_metric.num_nF > 0 ) {
    *fl = glnf_metric.nF[0];
    return TRUE;
  }

  return FALSE;

}



void introduce_minus_fluent( int fl )

{

  if ( gnum_relevant_fluents == MAX_RELEVANT_FLUENTS ) {
    printf("\ntoo many relevant fluents! increase MAX_RELEVANT_FLUENTS (currently %d)\n\n",
	   MAX_RELEVANT_FLUENTS);
    exit( 1 );
  }
  grelevant_fluents[gnum_relevant_fluents].function = -1;
  grelevant_fluents_name[gnum_relevant_fluents] = 
    ( char * ) calloc( MAX_LENGTH, sizeof( char ) );
  strcpy( grelevant_fluents_name[gnum_relevant_fluents], "MINUS-" );
  strcat( grelevant_fluents_name[gnum_relevant_fluents],
	  grelevant_fluents_name[fl] );
  lminus_fluent[fl] = gnum_relevant_fluents;
  lminus_fluent[gnum_relevant_fluents] = fl;
  gnum_relevant_fluents++;

}



void replace_fl_in_nF_with_minus_fl( int fl )

{

  Action *a;
  ActionEffect *e;
  int i, j, k, l;

  for ( i = 0; i < gnum_lnf_goal; i++ ) {
    for ( j = 0; j < glnf_goal_lh[i]->num_nF; j++ ) {
      if ( glnf_goal_lh[i]->nF[j] == fl ) break;
    }
    if ( j == glnf_goal_lh[i]->num_nF ) continue;
    /* now the jth fluent in subtraction is our translated one.
     *
     * first, put minus-fl into pF. Can't already be there
     * because we have only just introduced it.
     */
    if ( glnf_goal_lh[i]->num_pF == MAX_LNF_F ) {
      printf("\n\nincrease MAX_LNF_F! currently %d\n\n", MAX_LNF_F);
      exit( 1 );
    }
    glnf_goal_lh[i]->pF[glnf_goal_lh[i]->num_pF] = lminus_fluent[fl];
    glnf_goal_lh[i]->pC[glnf_goal_lh[i]->num_pF++] = glnf_goal_lh[i]->nC[j];
    /* now remove fl from nF.
     */
    for ( k = j; k < glnf_goal_lh[i]->num_nF - 1; k++ ) {
      glnf_goal_lh[i]->nF[k] = glnf_goal_lh[i]->nF[k+1];
      glnf_goal_lh[i]->nC[k] = glnf_goal_lh[i]->nC[k+1];
    }
    glnf_goal_lh[i]->num_nF--;
  }

  for ( a = gactions; a; a = a->next ) {
    for ( i = 0; i < a->num_lnf_preconds; i++ ) {
      for ( j = 0; j < a->lnf_preconds_lh[i]->num_nF; j++ ) {
	if ( a->lnf_preconds_lh[i]->nF[j] == fl ) break;
      }
      if ( j == a->lnf_preconds_lh[i]->num_nF ) continue;
      if ( a->lnf_preconds_lh[i]->num_pF == MAX_LNF_F ) {
	printf("\n\nincrease MAX_LNF_F! currently %d\n\n", MAX_LNF_F);
	exit( 1 );
      }
      a->lnf_preconds_lh[i]->pF[a->lnf_preconds_lh[i]->num_pF] = lminus_fluent[fl];
      a->lnf_preconds_lh[i]->pC[a->lnf_preconds_lh[i]->num_pF++] = a->lnf_preconds_lh[i]->nC[j];
      for ( k = j; k < a->lnf_preconds_lh[i]->num_nF - 1; k++ ) {
	a->lnf_preconds_lh[i]->nF[k] = a->lnf_preconds_lh[i]->nF[k+1];
	a->lnf_preconds_lh[i]->nC[k] = a->lnf_preconds_lh[i]->nC[k+1];
      }
      a->lnf_preconds_lh[i]->num_nF--;
    }

    for ( i = 0; i < a->num_effects; i++ ) {
      e = &(a->effects[i]);

      for ( j = 0; j < e->num_lnf_conditions; j++ ) {
	for ( k = 0; k < e->lnf_conditions_lh[j]->num_nF; k++ ) {
	  if ( e->lnf_conditions_lh[j]->nF[k] == fl ) break;
	}
	if ( k == e->lnf_conditions_lh[j]->num_nF ) continue;
	if ( e->lnf_conditions_lh[j]->num_pF == MAX_LNF_F ) {
	  printf("\n\nincrease MAX_LNF_F! currently %d\n\n", MAX_LNF_F);
	  exit( 1 );
	}
	e->lnf_conditions_lh[j]->pF[e->lnf_conditions_lh[j]->num_pF] = lminus_fluent[fl];
	e->lnf_conditions_lh[j]->pC[e->lnf_conditions_lh[j]->num_pF++] = e->lnf_conditions_lh[j]->nC[k];
	for ( l = k; l < e->lnf_conditions_lh[j]->num_nF - 1; l++ ) {
	  e->lnf_conditions_lh[j]->nF[l] = e->lnf_conditions_lh[j]->nF[l+1];
	  e->lnf_conditions_lh[j]->nC[l] = e->lnf_conditions_lh[j]->nC[l+1];
	}
	e->lnf_conditions_lh[j]->num_nF--;
      }

      if ( e->illegal ) {
	/* like before, we don't care about effects that access
	 * irrelevant fluents
	 */
	continue;
      }

      for ( j = 0; j < e->num_lnf_effects; j++ ) {
	for ( k = 0; k < e->lnf_effects_rh[j]->num_nF; k++ ) {
	  if ( e->lnf_effects_rh[j]->nF[k] == fl ) break;
	}
	if ( k == e->lnf_effects_rh[j]->num_nF ) continue;
	if ( e->lnf_effects_rh[j]->num_pF == MAX_LNF_F ) {
	  printf("\n\nincrease MAX_LNF_F! currently %d\n\n", MAX_LNF_F);
	  exit( 1 );
	}
	e->lnf_effects_rh[j]->pF[e->lnf_effects_rh[j]->num_pF] = lminus_fluent[fl];
	e->lnf_effects_rh[j]->pC[e->lnf_effects_rh[j]->num_pF++] = e->lnf_effects_rh[j]->nC[k];
	for ( l = k; l < e->lnf_effects_rh[j]->num_nF - 1; l++ ) {
	  e->lnf_effects_rh[j]->nF[l] = e->lnf_effects_rh[j]->nF[l+1];
	  e->lnf_effects_rh[j]->nC[l] = e->lnf_effects_rh[j]->nC[l+1];
	}
	e->lnf_effects_rh[j]->num_nF--;
      }
    }
  }

  for ( j = 0; j < glnf_metric.num_nF; j++ ) {
    if ( glnf_metric.nF[j] == fl ) break;
  }
  if ( j < glnf_metric.num_nF ) {
    if ( glnf_metric.num_pF == MAX_LNF_F ) {
      printf("\n\nincrease MAX_LNF_F! currently %d\n\n", MAX_LNF_F);
      exit( 1 );
    }
    glnf_metric.pF[glnf_metric.num_pF] = lminus_fluent[fl];
    glnf_metric.pC[glnf_metric.num_pF++] = glnf_metric.nC[j];
    for ( k = j; k < glnf_metric.num_nF - 1; k++ ) {
      glnf_metric.nF[k] = glnf_metric.nF[k+1];
      glnf_metric.nC[k] = glnf_metric.nC[k+1];
    }
    glnf_metric.num_nF--;
  }

}



void set_minus_fl_initial( int fl )

{

  if ( ginitial_state.f_D[fl] ) {
    ginitial_state.f_D[lminus_fluent[fl]] = TRUE;
    ginitial_state.f_V[lminus_fluent[fl]] =  (-1) * ginitial_state.f_V[fl];
  }

}



void introduce_minus_fl_effects( int fl )

{

  Action *a;
  ActionEffect *e;
  int i, j, k, pf, nf;
  LnfExpNode *len;

  for ( a = gactions; a; a = a->next ) {
    for ( i = 0; i < a->num_effects; i++ ) {
      e = &(a->effects[i]);
      if ( e->illegal ) {
	/* no need to translate illegal effects.
	 */
	continue;
      }

      for ( j = 0; j < e->num_lnf_effects; j++ ) {
	if ( e->lnf_effects_fl[j] != fl ) {
	  continue;
	}
	/* here is an effect that affects our fl.
	 * introduce inverse effect for minus_fl,
	 * making use of all minus-fl's that are already
	 * there.
	 */
	if ( e->num_lnf_effects == MAX_LNF_EFFS ) {
	  printf("\n\nincrease MAX_LNF_EFFS! currently %d\n\n", MAX_LNF_EFFS);
	  exit( 1 );
	}
	e->lnf_effects_neft[e->num_lnf_effects] = e->lnf_effects_neft[j];
	e->lnf_effects_fl[e->num_lnf_effects] = lminus_fluent[fl];
	e->lnf_effects_rh[e->num_lnf_effects] = new_LnfExpNode();
	len = e->lnf_effects_rh[e->num_lnf_effects];
	/* now the most "difficult" part: setup the inverted pF and nF
	 * informations.
	 *
	 * NOTE: as fluent occurences are unique in original ef,
	 *       so will they be in new ef. (no len contains both
	 *       a fluent and its minus-fluent)
	 *       --> invariant is or should be that the absolute
	 *           fluents occur at most once in |pF| \cup |nF|.
	 *           holds in the beginning.  only thing we do is
	 *           we exchange in that set for some fluents the
	 *           positive with the negative version, so the
	 *           invariant is in fact preserved.
	 */
	for ( k = 0; k < e->lnf_effects_rh[j]->num_pF; k++ ) {
	  pf = e->lnf_effects_rh[j]->pF[k];
	  if ( lminus_fluent[pf] == -1 ) {
	    /* not translated yet --> insert it into nF
	     */
	    if ( len->num_nF == MAX_LNF_F ) {
	      printf("\n\nincrease MAX_LNF_F! currently %d\n\n", MAX_LNF_F);
	      exit( 1 );
	    }
	    len->nF[len->num_nF] = pf;
	    len->nC[len->num_nF++] = e->lnf_effects_rh[j]->pC[k];
	  } else {
	    /* else, insert minus-pf into pF
	     */
	    if ( len->num_pF == MAX_LNF_F ) {
	      printf("\n\nincrease MAX_LNF_F! currently %d\n\n", MAX_LNF_F);
	      exit( 1 );
	    }
	    len->pF[len->num_pF] = lminus_fluent[pf];
	    len->pC[len->num_pF++] = e->lnf_effects_rh[j]->pC[k];
	  }
	}
	for ( k = 0; k < e->lnf_effects_rh[j]->num_nF; k++ ) {
	  nf = e->lnf_effects_rh[j]->nF[k];
	  /* insert all of those into pF
	   */
	  if ( len->num_pF == MAX_LNF_F ) {
	    printf("\n\nincrease MAX_LNF_F! currently %d\n\n", MAX_LNF_F);
	    exit( 1 );
	  }
	  len->pF[len->num_pF] = nf;
	  len->pC[len->num_pF++] = e->lnf_effects_rh[j]->nC[k];
	}
	/* the constant must of course be inverted.
	 */
	len->c = (-1) * e->lnf_effects_rh[j]->c;
	e->num_lnf_effects++;
      }
    }
  }

}


















/*************************************************************
 * LNF POST-PROCESSING I: SUMMARIZE EFFECTS.
 *************************************************************/



















int *lA, *lD;
int lnum_A, lnum_D;






void summarize_effects( void )

{

  Action *a;
  ActionEffect *e, *e_;
  int i, j, k, l;

  lA = ( int * ) calloc( gnum_relevant_facts, sizeof( int ) );
  lD = ( int * ) calloc( gnum_relevant_facts, sizeof( int ) );

  for ( a = gactions; a; a = a->next ) {
    i = 0;
    while ( i < a->num_effects ) {
      e = &(a->effects[i]);
      if ( e->removed ) {
	/* this one's already handled.
	 */
	i++;
	continue;
      }

      /* first, merge the effect's own effects together. logical:
       */
      lnum_A = 0;
      for ( j = 0; j < e->num_adds; j++ ) {
	for ( k = 0; k < lnum_A; k++ ) {
	  if ( lA[k] == e->adds[j] ) break;
	}
	if ( k < lnum_A ) continue;
	lA[lnum_A++] = e->adds[j];
      }
      lnum_D = 0;
      for ( j = 0; j < e->num_dels; j++ ) {
	for ( k = 0; k < lnum_D; k++ ) {
	  if ( lD[k] == e->dels[j] ) break;
	}
	if ( k < lnum_D ) continue;
	lD[lnum_D++] = e->dels[j];
      }
      /* numerical:
       */
      j = 0;
      while ( j < e->num_lnf_effects ) {
	/* merge all effects increasing the same fluent into
	 * effect j, and remove them.
	 */
	k = j + 1;
	while ( k < e->num_lnf_effects ) {
	  if ( e->lnf_effects_fl[k] != e->lnf_effects_fl[j] ) {
	    k++;
	    continue;
	  }
	  if ( e->lnf_effects_neft[j] == ASSIGN ) {
	    if ( e->lnf_effects_neft[k] != ASSIGN ||
		 !same_lnfs( e->lnf_effects_rh[j], e->lnf_effects_rh[k] ) ) {
	      e->illegal = TRUE;
	      break;
	    }
	  } else {
	    if ( e->lnf_effects_neft[k] == ASSIGN ) {
	      e->illegal = TRUE;
	      break;
	    }
	    merge_lnfs( e->lnf_effects_rh[j], e->lnf_effects_rh[k] );
	  }
	  /* we also get here if we have two identical assigns.
	   */
	  free( e->lnf_effects_rh[k] );
	  for ( l = k; l < e->num_lnf_effects - 1; l++ ) {
	    e->lnf_effects_neft[l] = e->lnf_effects_neft[l+1];
	    e->lnf_effects_fl[l] = e->lnf_effects_fl[l+1];
	    e->lnf_effects_rh[l] = e->lnf_effects_rh[l+1];
	  }
	  e->num_lnf_effects--;
	}
	if ( k < e->num_lnf_effects ) {
	  /* illegal combination
	   */
	  break;
	}
	j++;
      }

      /* now merge all effects after i with same condition
       * into that.
       */
      j = i + 1;
      while ( j < a->num_effects ) {
	e_ = &(a->effects[j]);
	if ( e_->removed ) {
	  j++;
	  continue;
	}

	if ( !same_condition( e, e_ ) ) {
	  j++;
	  continue;
	}
	/* no matter what happens, we can get rid of effect e_
	 */
	e_->removed = TRUE;

	/* illegality is inherited in both directions.
	 */
	if ( e_->illegal ) {
	  e->illegal = TRUE;
	}
	if ( e->illegal ) {
	  /* just for docu; it is removed anyway.
	   */
	  e_->illegal = TRUE;
	}

	if ( !e->illegal ) {
	  /* the combined effect appears to be legal. merge it.
	   */
	  merge_effects( e, e_ );
	  if ( e->illegal ) {
	    /* e might have become illegal. again, docu this.
	     */
	    e_->illegal = TRUE;
	  }
	}

	j++;
      }

      /* now put the updated A and D info into e.
       *
       * have to be careful: it might be that there are
       * now too many facts and we need to re-allocate
       * e's capabilities.
       */
      if ( lnum_A > e->num_adds ) {
	free( e->adds );
	e->adds = ( int * ) calloc( lnum_A, sizeof( int ) );
      }
      for ( j = 0; j < lnum_A; j++ ) {
	e->adds[j] = lA[j];
      }
      e->num_adds = lnum_A;
      if ( lnum_D > e->num_dels ) {
	free( e->dels );
	e->dels = ( int * ) calloc( lnum_D, sizeof( int ) );
      }
      for ( j = 0; j < lnum_D; j++ ) {
	e->dels[j] = lD[j];
      }
      e->num_dels = lnum_D;

      /* increment current effects counter.
       */
      i++;
    }
  }

}



Bool same_condition( ActionEffect *e, ActionEffect *e_ )

{

  int i, j;

  if ( e->num_conditions != e_->num_conditions ||
       e->num_lnf_conditions != e_->num_lnf_conditions ) return FALSE;

  for ( i = 0; i < e->num_conditions; i++ ) {
    for ( j = 0; j < e_->num_conditions; j++ ) {
      if ( e->conditions[i] == e_->conditions[j] ) break;
    }
    if ( j == e_->num_conditions ) break;
  }
  if ( i < e->num_conditions ) return FALSE;

  for ( i = 0; i < e->num_lnf_conditions; i++ ) {
    for ( j = 0; j < e_->num_lnf_conditions; j++ ) {
      if ( e_->lnf_conditions_comp[j] != e->lnf_conditions_comp[i] ) continue;
      if ( e_->lnf_conditions_rh[j] != e->lnf_conditions_rh[i] ) continue;
      if ( !same_lnfs( e_->lnf_conditions_lh[j], e->lnf_conditions_lh[i] ) ) continue;
      break;
    }
    if ( j == e_->num_lnf_conditions ) break;
  }
  if ( i < e->num_lnf_conditions ) return FALSE;

  return TRUE;

}



Bool same_lnfs( LnfExpNode *l, LnfExpNode *r )

{

  int i, j;

  if ( l->num_pF != r->num_pF ||
       l->c != r->c ) return FALSE;

  for ( i = 0; i < l->num_pF; i++ ) {
    for ( j = 0; j < r->num_pF; j++ ) {
      if ( l->pF[i] != r->pF[j] ) continue;
      if ( l->pC[i] != r->pC[j] ) {
	/* same fluent with different weighting.
	 */
	return FALSE;
      }
      break;
    }
    if ( j == r->num_pF ) break;
  }
  if ( i < l->num_pF ) return FALSE;

  return TRUE;

}



void merge_effects( ActionEffect *e, ActionEffect *e_ )

{

  int i, j;

  /* we don't care whether adds and dels intersect:
   * they're allowed to by semantics.
   */
  for ( i = 0; i < e_->num_adds; i++ ) {
    for ( j = 0; j < lnum_A; j++ ) {
      if ( lA[j] == e_->adds[i] ) break;
    }
    if ( j < lnum_A ) continue;
    lA[lnum_A++] = e_->adds[i];
  }
  for ( i = 0; i < e_->num_dels; i++ ) {
    for ( j = 0; j < lnum_D; j++ ) {
      if ( lD[j] == e_->dels[i] ) break;
    }
    if ( j < lnum_D ) continue;
    lD[lnum_D++] = e_->dels[i];
  }

  for ( i = 0; i < e_->num_lnf_effects; i++ ) {
    for ( j = 0; j < e->num_lnf_effects; j++ ) {
      if ( e->lnf_effects_fl[j] == e_->lnf_effects_fl[i] ) break;
    }
    if ( j == e->num_lnf_effects ) {
      /* new affected fluent!
       */
      if ( e->num_lnf_effects == MAX_LNF_EFFS ) {
	printf("\n\nincrease MAX_LNF_EFFS! currently %d\n\n", MAX_LNF_EFFS);
	exit( 1 );
      }
      e->lnf_effects_neft[e->num_lnf_effects] = e_->lnf_effects_neft[i];
      e->lnf_effects_fl[e->num_lnf_effects] = e_->lnf_effects_fl[i];
      /* we can also simply take the pointer: e_ is only marked as removed,
       * but not freed.
       */
      e->lnf_effects_rh[e->num_lnf_effects] = e_->lnf_effects_rh[i];
      e->num_lnf_effects++;
    } else {
      if ( e->lnf_effects_neft[j] == ASSIGN ) {
	if ( e_->lnf_effects_neft[i] != ASSIGN ||
	     !same_lnfs( e->lnf_effects_rh[j], e_->lnf_effects_rh[i] ) ) {
	  e->illegal = TRUE;
	  return;
	}
	/* identical assigns. nothing needs to be done.
	 */
      } else {
	if ( e_->lnf_effects_neft[i] == ASSIGN ) {
	  e->illegal = TRUE;
	  return;
	}
	merge_lnfs( e->lnf_effects_rh[j], e_->lnf_effects_rh[i] );
      }
    }
  }

}



/* merge both LNFs into the left one.
 * (only pF needed as both are already 
 * fully transformed)
 */
void merge_lnfs( LnfExpNode *l, LnfExpNode *r )

{

  int i, j, k;

  for ( i = 0; i < r->num_pF; i++ ) {

    for ( j = 0; j < l->num_pF; j++ ) {
      if ( r->pF[i] == l->pF[j] ) break;
    }
    if ( j < l->num_pF ) {
      /* got that one in dest LNF already
       */
      l->pC[j] += r->pC[i];
      continue;
    }

    if ( lminus_fluent[r->pF[i]] != -1 ) {
      /* this one was already translated. let's see
       * if its counterpart is in the left lnf.
       */
      for ( j = 0; j < l->num_pF; j++ ) {
	if ( lminus_fluent[r->pF[i]] == l->pF[j] ) break;
      }
      if ( j < l->num_pF ) {
	/* for this, we got the inverse one!
	 */
	l->pC[j] -= r->pC[i];
	if ( l->pC[j] < 0 ) {
	  l->pF[j] = r->pF[i];
	  l->pC[j] *= (-1);
	}
	if ( l->pC[j] == 0 ) {
	  /* remove this entirely.
	   */
	  for ( k = j; k < l->num_pF - 1; k++ ) {
	    l->pF[k] = l->pF[k+1];
	    l->pC[k] = l->pC[k+1];
	  }
	  l->num_pF--;
	}
	continue;
      }
    }

    /* we got neither that nor its counterpart.
     */
    if ( l->num_pF == MAX_LNF_F ) {
      printf("\n\nincrease MAX_LNF_F! currently %d\n\n", MAX_LNF_F);
      exit( 1 );
    }
    l->pF[l->num_pF] = r->pF[i];
    l->pC[l->num_pF++] = r->pC[i];
  }


  l->c += r->c;

}






















/*************************************************************
 * LNF POST-PROCESSING II: ENCODE NON-MINIMAL LNFs.
 *************************************************************/























void encode_lfns_as_artificial_fluents( void )

{

  int i;

  /* for the artificial new ones, this will be set
   * to the respective LNF.
   */
  for ( i = 0; i < MAX_RELEVANT_FLUENTS; i++ ) {
    grelevant_fluents_lnf[i] = NULL;
  }

  while ( TRUE ) {
    /* ex non-minimal lnf in pre, cond, eff, or goal?
     *
     * (i.e., lnf != fl + c)
     */
    if ( !ex_non_minimal_lnf_in_pre_cond_goal_eff() ) {
      /* no --> we are finished.
       */
      break;
    }
    /* otherwise, the respective LNF, without the 
     * constant part, is set up in
     * lF...; (local global borrowed from above);
     *
     * introduce a new artificial fluent for that
     * LNF
     */
    introduce_artificial_fluent();
    /* replace all occurences in pres, conds, effs, and goals
     */
    replace_non_minimal_lnf_with_artificial_fl();
  }

}



Bool ex_non_minimal_lnf_in_pre_cond_goal_eff( void )

{

  Action *a;
  ActionEffect *e;
  int i, j, k;

  for ( i = 0; i < gnum_lnf_goal; i++ ) {
    if ( glnf_goal_lh[i]->num_pF > 1 ||
	 (glnf_goal_lh[i]->num_pF == 1 && glnf_goal_lh[i]->pC[0] != 1) ) {
      for ( j = 0; j < glnf_goal_lh[i]->num_pF; j++ ) {
	lF[j] = glnf_goal_lh[i]->pF[j];
	lC[j] = glnf_goal_lh[i]->pC[j];
      }
      lnum_F = glnf_goal_lh[i]->num_pF;
      return TRUE;
    }
  }

  for ( a = gactions; a; a = a->next ) {
    for ( i = 0; i < a->num_lnf_preconds; i++ ) {
      if ( a->lnf_preconds_lh[i]->num_pF > 1 ||
	   (a->lnf_preconds_lh[i]->num_pF == 1 && a->lnf_preconds_lh[i]->pC[0] != 1) ) {
	for ( j = 0; j < a->lnf_preconds_lh[i]->num_pF; j++ ) {
	  lF[j] = a->lnf_preconds_lh[i]->pF[j];
	  lC[j] = a->lnf_preconds_lh[i]->pC[j];
	}
	lnum_F = a->lnf_preconds_lh[i]->num_pF;
	return TRUE;
      }
    }

    for ( i = 0; i < a->num_effects; i++ ) {
      e = &(a->effects[i]);
      if ( e->removed ) {
	/* these will not be included into conn:
	 * merged into somewhere else.
	 */
	continue;
      }

      for ( j = 0; j < e->num_lnf_conditions; j++ ) {
	if ( e->lnf_conditions_lh[j]->num_pF > 1 ||
	     (e->lnf_conditions_lh[j]->num_pF == 1 && e->lnf_conditions_lh[j]->pC[0] != 1) ) {
	  for ( k = 0; k < e->lnf_conditions_lh[j]->num_pF; k++ ) {
	    lF[k] = e->lnf_conditions_lh[j]->pF[k];
	    lC[k] = e->lnf_conditions_lh[j]->pC[k];
	  }
	  lnum_F = e->lnf_conditions_lh[j]->num_pF;
	  return TRUE;
	}
      }

      if ( e->illegal ) {
	continue;
      }

      for ( j = 0; j < e->num_lnf_effects; j++ ) {
	if ( e->lnf_effects_rh[j]->num_pF > 1 ||
	     (e->lnf_effects_rh[j]->num_pF == 1 && e->lnf_effects_rh[j]->pC[0] != 1) ) {
	  for ( k = 0; k < e->lnf_effects_rh[j]->num_pF; k++ ) {
	    lF[k] = e->lnf_effects_rh[j]->pF[k];
	    lC[k] = e->lnf_effects_rh[j]->pC[k];
	  }
	  lnum_F = e->lnf_effects_rh[j]->num_pF;
	  return TRUE;
	}
      }
    }
  }

  return FALSE;

}



void introduce_artificial_fluent( void )

{

  int i;

  if ( gnum_relevant_fluents == MAX_RELEVANT_FLUENTS ) {
    printf("\ntoo many relevant fluents! increase MAX_RELEVANT_FLUENTS (currently %d)\n\n",
	   MAX_RELEVANT_FLUENTS);
    exit( 1 );
  }
  grelevant_fluents[gnum_relevant_fluents].function = -1;

  /* no name --> is inferred in this case from _lnf
   */

  grelevant_fluents_lnf[gnum_relevant_fluents] = new_LnfExpNode();
  for ( i = 0; i < lnum_F; i++ ) {
    grelevant_fluents_lnf[gnum_relevant_fluents]->pF[i] = lF[i];
    grelevant_fluents_lnf[gnum_relevant_fluents]->pC[i] = lC[i];
  }
  grelevant_fluents_lnf[gnum_relevant_fluents]->num_pF = lnum_F;

  gnum_relevant_fluents++;

}



void replace_non_minimal_lnf_with_artificial_fl( void )

{

  Action *a;
  ActionEffect *e;
  int i, j;

  for ( i = 0; i < gnum_lnf_goal; i++ ) {
    if ( !is_artificial_fluent( glnf_goal_lh[i] ) ) {
      continue;
    }
    /* the pF here is the pF we are currently replacing.
     */
    glnf_goal_lh[i]->pF[0] = gnum_relevant_fluents - 1;
    glnf_goal_lh[i]->pC[0] = 1;
    glnf_goal_lh[i]->num_pF = 1;
  }

  for ( a = gactions; a; a = a->next ) {
    for ( i = 0; i < a->num_lnf_preconds; i++ ) {
      if ( !is_artificial_fluent( a->lnf_preconds_lh[i] ) ) {
	continue;
      }
      a->lnf_preconds_lh[i]->pF[0] = gnum_relevant_fluents - 1;
      a->lnf_preconds_lh[i]->pC[0] = 1;
      a->lnf_preconds_lh[i]->num_pF = 1;
    }

    for ( i = 0; i < a->num_effects; i++ ) {
      e = &(a->effects[i]);
      if ( e->removed ) {
	/* these will not be included into conn:
	 * merged into somewhere else.
	 */
	continue;
      }

      for ( j = 0; j < e->num_lnf_conditions; j++ ) {
	if ( !is_artificial_fluent( e->lnf_conditions_lh[j] ) ) {
	  continue;
	}
	e->lnf_conditions_lh[j]->pF[0] = gnum_relevant_fluents - 1;
	e->lnf_conditions_lh[j]->pC[0] = 1;
	e->lnf_conditions_lh[j]->num_pF = 1;
      }

      if ( e->illegal ) {
	continue;
      }

      for ( j = 0; j < e->num_lnf_effects; j++ ) {
	if ( !is_artificial_fluent( e->lnf_effects_rh[j] ) ) {
	  continue;
	}
	e->lnf_effects_rh[j]->pF[0] = gnum_relevant_fluents - 1;
	e->lnf_effects_rh[j]->pC[0] = 1;
	e->lnf_effects_rh[j]->num_pF = 1;
      }
    }
  }

}



Bool is_artificial_fluent( LnfExpNode *n )

{

  int i, j;

  if ( n->num_nF != 0 ) {
    printf("\n\nchecking non-empty nF for multiple fl!\n\n");
    exit( 1 );
  }

  if ( n->num_pF != lnum_F ) {
    return FALSE;
  }

  for ( i = 0; i < lnum_F; i++ ) {
    for ( j = 0; j < n->num_pF; j++ ) {
      if ( n->pF[j] != lF[i] ) continue;
      if ( n->pC[j] != lC[i] ) {
	/* wrong constant multiplier!
	 */
	return FALSE;
      }
      break;
    }
    if ( j == n->num_pF ) {
      /* didn't find this fluent i in here.
       */
      return FALSE;
    }
  }

  return TRUE;

}


















/*************************************************************
 * AT LAST: PREPARATIONS FOR METRIC FUNCTION
 *************************************************************/


















Bool setup_effect_costs( void )

{

  Action *a;
  ActionEffect *e;
  int i, j, k, fl;
  Bool non_zero = FALSE;

  if ( glnf_metric.num_pF == 0 ) {
    /* no metric, or previously failed
     */
    if ( gcmd_line.display_info ) {
      printf("\nno metric specified.");
    }
    return FALSE;
  }

  /* also in here: check if all parts of metric are defined
   * if not, then they won't ever be because we do not allow
   * assigners anyway. 
   *
   * also, setup gtt total-time multipl. 
   * currently needed since in h fn effect cists are summed up
   * --> may count the same action more than once, if we insert the
   * timing cost into the effect cost.
   *
   * ... this is AWKWARD... probably would be better to simply 
   * associate costs always (including relaxed plans)
   * only with ACTIONS!
   */
  gtt = 0;
  for ( i = 0; i < glnf_metric.num_pF; i++ ) {
    if ( glnf_metric.pF[i] == -2 ) {
      gtt = glnf_metric.pC[i];
      continue;
    }
    if ( !ginitial_state.f_D[glnf_metric.pF[i]] ) break;
  }
  if ( i < glnf_metric.num_pF ) {
    if ( gcmd_line.display_info ) {
      printf("\nwarning: metric undefined initially. no optimization done.");
    }
    return FALSE;
  }

  for ( a = gactions; a; a = a->next ) {
    for ( i = 0; i < a->num_effects; i++ ) {
      e = &(a->effects[i]);
      e->cost = 0;

      if ( e->removed ||
	   e->illegal ) {
	continue;
      }

      for ( j = 0; j < e->num_lnf_effects; j++ ) {
	fl = e->lnf_effects_fl[j];
	for ( k = 0; k < glnf_metric.num_pF; k++ ) {
	  if ( fl == glnf_metric.pF[k] ) break;
	}
	if ( k == glnf_metric.num_pF ) continue;

	if ( e->lnf_effects_rh[j]->num_pF > 0 ) {
	  if ( gcmd_line.display_info ) {
	    printf("\nwarning: non-constant effect on metric. no optimization done.");
	  }
	  return FALSE;
	}
	if ( e->lnf_effects_neft[j] != INCREASE ) {
	  if ( gcmd_line.display_info ) {
	    printf("\nwarning: assign on metric. no optimization done.");
	  }
	  return FALSE;
	}
	if ( e->lnf_effects_rh[j]->c < 0 ) {
	  if ( gcmd_line.display_info ) {
	    printf("\nwarning: change on metric in wrong direction. no optimization done.");
	  }
	  return FALSE;
	}

	e->cost += glnf_metric.pC[k] * e->lnf_effects_rh[j]->c;
	if ( e->cost > 0 ) {
	  non_zero = TRUE;
	}
      }
    }
  }

  if ( !non_zero ) {
    if ( gtt == 0 ) {
      if ( gcmd_line.display_info ) {
	printf("\nwarning: trivial metric, all costs 0. no optimization done.");
      }
      return FALSE;
    }
  }

  return TRUE;

}





















/*************************************************************
 * AT VERY LAST: ACYCLIC := EFFS, AND STATIC FL RELEVANCE
 *************************************************************/























void check_assigncycles( void )

{

  int i, j, k, c = 0;

  gassign_influence = ( Bool ** ) calloc( gnum_real_fl_conn, sizeof( Bool* ) );
  gTassign_influence = ( Bool ** ) calloc( gnum_real_fl_conn, sizeof( Bool* ) );
  for ( i = 0; i < gnum_real_fl_conn; i++ ) {
    gassign_influence[i] = ( Bool * ) calloc( gnum_real_fl_conn, sizeof( Bool ) );
    gTassign_influence[i] = ( Bool * ) calloc( gnum_real_fl_conn, sizeof( Bool ) );
  }

  if ( gcmd_line.display_info > 1 ) {
    printf("\n\nchecking for cyclic := effects");
  }
  for ( i = 0; i < gnum_real_fl_conn; i++ ) {
    for ( j = 0; j < gnum_real_fl_conn; j++ ) {
      gassign_influence[i][j] = i_influences_j( i, j );
      gTassign_influence[i][j] = i_influences_j( i, j );
    }
  }
  /* compute transitive closure on dependencies
   */
  for ( j = 0; j < gnum_real_fl_conn; j++ ) {
    for ( i = 0; i < gnum_real_fl_conn; i++ ) {
      if ( gTassign_influence[i][j] ) {
	for ( k = 0; k < gnum_real_fl_conn; k++ ) {
	  if ( gTassign_influence[j][k] ) {
	    gTassign_influence[i][k] = TRUE;
	  }
	}
      }
    }
  }
  for ( i = 0; i < gnum_real_fl_conn; i++ ) {
    if ( gTassign_influence[i][i] ) {
      printf("\nnumerical variable ");
      print_fl_name( i );
      printf(" lies on := propagation cycle!");
      c++;
    }
  }
  if ( c > 0 ) {
    printf("\nexit. (mneed computation not possible, RPG termination unclear)");
    printf("\n                (questions to Joerg Hoffmann)\n\n");
    exit( 1 );
  } else {
    if ( gcmd_line.display_info > 1 ) {
      printf(" --- OK.");
    }
  }

}



Bool i_influences_j( int fi, int fj )

{

  int i, j, fl_;

  for ( i = 0; i < gfl_conn[fj].num_AS; i++ ) {
    fl_ = gfl_conn[fj].AS_fl_[i];
    if ( fl_ < 0 ) continue;
    if ( fl_ == fi ) return TRUE;
    if ( !gfl_conn[fl_].artificial ) continue;
    for ( j = 0; j < gfl_conn[fl_].num_lnf; j++ ) {
      if ( gfl_conn[fl_].lnf_F[j] == fi ) return TRUE;
    }
  }

  return FALSE;

}



void determine_fl_relevance( void )

{

  int i, j, k, fl, fl_, ef, pc, g;
  Bool **influenced_by;

  /* this here contains transfers from i to j i.e. if
   * i is relevant then j is too
   */
  influenced_by = ( Bool ** ) calloc( gnum_real_fl_conn, sizeof( Bool* ) );
  for ( i = 0; i < gnum_real_fl_conn; i++ ) {
    influenced_by[i] = ( Bool * ) calloc( gnum_real_fl_conn, sizeof( Bool ) );
  }
  for ( i = 0; i < gnum_real_fl_conn; i++ ) {
    for ( j = 0; j < gnum_real_fl_conn; j++ ) {
      influenced_by[i][j] = ( gassign_influence[j][i] ||
			       i_inc_influences_j( j, i ) );
    }
  }
  /* transitive closure so we'll have direct access below.
   */
  for ( j = 0; j < gnum_real_fl_conn; j++ ) {
    for ( i = 0; i < gnum_real_fl_conn; i++ ) {
      if ( influenced_by[i][j] ) {
	for ( k = 0; k < gnum_real_fl_conn; k++ ) {
	  if ( influenced_by[j][k] ) {
	    influenced_by[i][k] = TRUE;
	  }
	}
      }
    }
  }

  for ( i = 0; i < gnum_real_fl_conn; i++ ) {
    gfl_conn[i].relevant = FALSE;
  }
  /* relevance originates in effect preconds and goals.
   */
  for ( ef = 0; ef < gnum_ef_conn; ef++ ) {
    for ( pc = 0; pc < gef_conn[ef].num_f_PC; pc++ ) {
      /* constraint here is gef_conn[ef].f_PC_fl[pc] >= [>] gef_conn[ef].f_PC_c[pc]
       * where lh side can be lnf expression.
       */
      fl = gef_conn[ef].f_PC_fl[pc];
      if ( fl < 0 ) {
	printf("\nnegative constr lh??\n\n");
	exit( 1 );
      }
      if ( !gfl_conn[fl].artificial ) {
	gfl_conn[fl].relevant = TRUE;
	for ( i = 0; i < gnum_real_fl_conn; i++ ) {
	  if ( influenced_by[fl][i] ) gfl_conn[i].relevant = TRUE;
	}
      } else {
	for ( i = 0; i < gfl_conn[fl].num_lnf; i++ ) {
	  fl_ = gfl_conn[fl].lnf_F[i];
	  gfl_conn[fl_].relevant = TRUE;
	  for ( j = 0; j < gnum_real_fl_conn; j++ ) {
	    if ( influenced_by[fl_][j] ) gfl_conn[j].relevant = TRUE;
	  }
	}
      }
    }
  }
  for ( g = 0; g < gnum_fnumeric_goal; g++ ) {
    /* constraint here is gfnumeric_goal_fl[g] >= [>] gfnumeric_goal_c[g]
     * where lh side can be lnf expression.
     */
    fl = gfnumeric_goal_fl[g];
    if ( fl < 0 ) {
      printf("\nnegative constr lh??\n\n");
      exit( 1 );
    }
    if ( !gfl_conn[fl].artificial ) {
      gfl_conn[fl].relevant = TRUE;
      for ( i = 0; i < gnum_real_fl_conn; i++ ) {
	if ( influenced_by[fl][i] ) gfl_conn[i].relevant = TRUE;
      }
    } else {
      for ( i = 0; i < gfl_conn[fl].num_lnf; i++ ) {
	fl_ = gfl_conn[fl].lnf_F[i];
	gfl_conn[fl_].relevant = TRUE;
	for ( j = 0; j < gnum_real_fl_conn; j++ ) {
	  if ( influenced_by[fl_][j] ) gfl_conn[j].relevant = TRUE;
	}
      }
    }
  }

  if ( 0 ) {
    for ( i = 0; i < gnum_real_fl_conn; i++ ) {
      printf("\n"); print_fl_name( i );
      printf (" --- relevant: %d", gfl_conn[i].relevant);
    }
  }

}



Bool i_inc_influences_j( int fi, int fj )

{

  int i, j, fl_;

  for ( i = 0; i < gfl_conn[fj].num_IN; i++ ) {
    fl_ = gfl_conn[fj].IN_fl_[i];
    if ( fl_ < 0 ) continue;
    if ( fl_ == fi ) return TRUE;
    if ( !gfl_conn[fl_].artificial ) continue;
    for ( j = 0; j < gfl_conn[fl_].num_lnf; j++ ) {
      if ( gfl_conn[fl_].lnf_F[j] == fi ) return TRUE;
    }
  }

  return FALSE;

}

