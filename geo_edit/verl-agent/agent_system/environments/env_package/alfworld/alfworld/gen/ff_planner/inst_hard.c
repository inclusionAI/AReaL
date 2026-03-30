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
 * File: inst_hard.c
 * Description: functions for multiplying hard operators.
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
#include "inst_hard.h" 











/* used in multiplying routines
 */
int linst_table[MAX_VARS];
int_pointer lini[MAX_PREDICATES];








void build_hard_action_templates( void )

{

  int i, j, size, adr;
  MixedOperator *o;

  /* remove unused params; empty types are already recognised during
   * domain translation; have to be handled after (or while)
   * unaries encoding (if done), though.
   */
  cleanup_hard_domain();

  if ( gcmd_line.display_info == 115 ) {
    printf("\n\ncleaned up hard domain representation is:\n\n");
    for ( i = 0; i < gnum_hard_operators; i++ ) {
      print_Operator( ghard_operators[i] );
    }
    fflush( stdout );
  }

  /* create local table of instantiated facts that occur in the
   * initial state. for fast finding out if fact is in ini or not.
   */
  for ( i = 0; i < gnum_predicates; i++ ) {
    size = 1;
    for ( j = 0; j < garity[i]; j++ ) {
      size *= gnum_constants;
    }
    lini[i] = ( int_pointer ) calloc( size, sizeof( int ) );
    for ( j = 0; j < size; j++ ) {
      lini[i][j] = 0;
    }
    for ( j = 0; j < gnum_initial_predicate[i]; j++ ) {
      adr = instantiated_fact_adress( &ginitial_predicate[i][j] );
      lini[i][adr]++;
    }
  }


  /* create mixed op for each param combination
   */
  multiply_hard_op_parameters();

  if ( gcmd_line.display_info == 116 ) {
    printf("\n\nmixed hard domain representation is:\n\n");
    for ( o = ghard_mixed_operators; o; o = o->next ) {
      print_MixedOperator( o );
    }
    fflush( stdout );
  }

  /* create pseudo op for each mixed op
   */
  multiply_hard_effect_parameters();
 
  if ( gcmd_line.display_info == 117 ) {
    printf("\n\npseudo hard domain representation is:\n\n");
    for ( i = 0; i < gnum_hard_templates; i++ ) {
      print_PseudoAction( ghard_templates[i] );
    }
    fflush( stdout );
  }
 

}












/****************
 * CLEANUP CODE *
 ****************/












void cleanup_hard_domain( void )

{

  int i, j, k, par;
  Operator *o;
  Effect *e;

  /* so far, only unused parameters removal
   */

  for ( i = 0; i < gnum_hard_operators; i++ ) {
    o = ghard_operators[i];

    j = 0;
    while ( j < o->num_vars ) {
      if ( var_used_in_wff( ENCODE_VAR( j ), o->preconds ) ) {
	j++;
	continue;
      }

      for ( e = o->effects; e; e = e->next ) {
	if ( var_used_in_wff( ENCODE_VAR( j ), e->conditions ) ) {
	  break;
	}
	if ( var_used_in_literals( ENCODE_VAR( j ), e->effects ) ) {
	  break;
	}
	if ( var_used_in_numeric_effects( ENCODE_VAR( j ), e->numeric_effects ) ) {
	  break;
	}
      }
      if ( e ) {
	j++;
	continue;
      }

      o->removed[j] = TRUE;
      j++;
    }

    for ( e = o->effects; e; e = e->next ) {
      j = 0;
      while ( j < e->num_vars ) {
	par = o->num_vars + j;
	if ( var_used_in_wff( ENCODE_VAR( par ), e->conditions ) ) {
	  j++;
	  continue;
	}
	if ( var_used_in_literals( ENCODE_VAR( par ), e->effects ) ) {
	  j++;
	  continue;
	}
	if ( var_used_in_numeric_effects( ENCODE_VAR( par ), e->numeric_effects ) ) {
	  j++;
	  continue;
	}

	if ( e->var_names[j] ) {
	  free( e->var_names[j] );
	}
	for ( k = j; k < e->num_vars - 1; k++ ) {
	  e->var_names[k] = e->var_names[k+1];
	  e->var_names[k] = e->var_names[k+1];
	}
	e->num_vars--;
	decrement_inferior_vars( par, e->conditions );
	decrement_inferior_vars_in_literals( par, e->effects );
	decrement_inferior_vars_in_numeric_effects( par, e->numeric_effects );
      }
    }
  }

}



Bool var_used_in_literals( int code_var, Literal *ef )

{

  Literal *l;
  int i;
  
  for ( l = ef; l; l = l->next ) {
    for ( i = 0; i < garity[l->fact.predicate]; i++ ) {
      if ( l->fact.args[i] == code_var ) {
	return TRUE;
      }
    }
  }

  return FALSE;

}



Bool var_used_in_numeric_effects( int code_var, NumericEffect *ef )

{

  NumericEffect *l;
  int i;
  
  for ( l = ef; l; l = l->next ) {
    for ( i = 0; i < gf_arity[l->fluent.function]; i++ ) {
      if ( l->fluent.args[i] == code_var ) {
	return TRUE;
      }
    }
    if ( var_used_in_exp( code_var, l->rh ) ) {
      return TRUE;
    }
  }

  return FALSE;

}



void decrement_inferior_vars_in_literals( int var, Literal *ef )

{

  Literal *l;
  int i;
  
  for ( l = ef; l; l = l->next ) {
    for ( i = 0; i < garity[l->fact.predicate]; i++ ) {
      if ( l->fact.args[i] >= 0 ) {
	continue;
      }
      if ( DECODE_VAR( l->fact.args[i] ) > var ) {
	l->fact.args[i]++;
      }
    }
  }

}



void decrement_inferior_vars_in_numeric_effects( int var, NumericEffect *ef )

{

  NumericEffect *l;
  int i;
  
  for ( l = ef; l; l = l->next ) {
    for ( i = 0; i < gf_arity[l->fluent.function]; i++ ) {
      if ( l->fluent.args[i] >= 0 ) {
	continue;
      }
      if ( DECODE_VAR( l->fluent.args[i] ) > var ) {
	l->fluent.args[i]++;
      }
    }
    decrement_inferior_vars_in_exp( var, l->rh );
  }

}














/******************************
 * CODE THAT BUILDS MIXED OPS *
 ******************************/














void multiply_hard_op_parameters( void )

{

  int i;

  ghard_mixed_operators = NULL;

  for ( i = 0; i < MAX_VARS; i++ ) {
    linst_table[i] = -1;
  }

  for ( i = 0; i < gnum_hard_operators; i++ ) {
    create_hard_mixed_operators( ghard_operators[i], 0 );
  }

}



void create_hard_mixed_operators( Operator *o, int curr_var )

{

  int t, i, m, mn;
  WffNode *tmp1, *w, *ww;
  MixedOperator *tmp2;

  if ( curr_var < o->num_vars ) {
    if ( o->removed[curr_var] ) {
      /* param doesn't matter -- select any appropriate type constant
       * at least one there; otherwise, op would not have been translated.
       */
      linst_table[curr_var] = gtype_consts[o->var_types[curr_var]][0];
      create_hard_mixed_operators( o, curr_var + 1 );
      linst_table[curr_var] = -1;
      return;
    }

    t = o->var_types[curr_var];
    for ( i = 0; i < gtype_size[t]; i++ ) {
      linst_table[curr_var] = gtype_consts[t][i];

      create_hard_mixed_operators( o, curr_var + 1 );

      linst_table[curr_var] = -1;
    }
    return;
  }


  tmp1 = instantiate_wff( o->preconds );

  if ( tmp1->connective == FAL ) {
    free_WffNode( tmp1 );
    return;
  }

  dnf( &tmp1 );
  cleanup_wff( &tmp1 );

  if ( tmp1->connective == FAL ) {
    free_WffNode( tmp1 );
    return;
  }

  /* only debugging, REMOVE LATER
   */
  if ( is_dnf( tmp1 ) == -1 ) {
    printf("\n\nILLEGAL DNF %s AFTER INSTANTIATION\n\n", o->name);
    print_Wff( tmp1, 0 );
    exit( 1 );
  }

  switch ( tmp1->connective ) {
  case OR:
    for ( w = tmp1->sons; w; w = w->next ) {
      tmp2 = new_MixedOperator( o );
      for ( i = 0; i < o->num_vars; i++ ) {
	tmp2->inst_table[i] = linst_table[i];
      }
      if ( w->connective == AND ) {
	m = 0;
	mn = 0;
	for ( ww = w->sons; ww; ww = ww->next ) {
	  if ( ww->connective == ATOM ) m++;
	  if ( ww->connective == COMP ) mn++;
	}
	tmp2->preconds = ( Fact * ) calloc( m, sizeof( Fact ) );
	tmp2->numeric_preconds_comp = ( Comparator * ) calloc( mn, sizeof( Comparator ) );
	tmp2->numeric_preconds_lh = ( ExpNode_pointer * ) calloc( mn, sizeof( ExpNode_pointer ) );
	tmp2->numeric_preconds_rh = ( ExpNode_pointer * ) calloc( mn, sizeof( ExpNode_pointer ) );
	tmp2->num_preconds = m;
	tmp2->num_numeric_preconds = mn;
	m = 0; mn = 0;
	for ( ww = w->sons; ww; ww = ww->next ) {
	  if ( ww->connective == ATOM ) {
	    tmp2->preconds[m].predicate = ww->fact->predicate;
	    for ( i = 0; i < garity[ww->fact->predicate]; i++ ) {
	      tmp2->preconds[m].args[i] = ww->fact->args[i];
	    }
	    m++;
	  }
	  if ( ww->connective == COMP ) {
	    tmp2->numeric_preconds_comp[mn] = ww->comp;
	    tmp2->numeric_preconds_lh[mn] = copy_Exp( ww->lh );
	    tmp2->numeric_preconds_rh[mn] = copy_Exp( ww->rh );
	    mn++;
	  }
	}
      } else {
	if ( w->connective == ATOM ) {
	  tmp2->preconds = ( Fact * ) calloc( 1, sizeof( Fact ) );
	  tmp2->num_preconds = 1;
	  tmp2->preconds[0].predicate = w->fact->predicate;
	  for ( i = 0; i < garity[w->fact->predicate]; i++ ) {
	    tmp2->preconds[0].args[i] = w->fact->args[i];
	  }
	}
	if ( w->connective == COMP ) {
	  tmp2->numeric_preconds_comp = ( Comparator * ) calloc( 1, sizeof( Comparator ) );
	  tmp2->numeric_preconds_lh = ( ExpNode_pointer * ) calloc( 1, sizeof( ExpNode_pointer ) );
	  tmp2->numeric_preconds_rh = ( ExpNode_pointer * ) calloc( 1, sizeof( ExpNode_pointer ) );
	  tmp2->numeric_preconds_comp[0] = w->comp;
	  tmp2->numeric_preconds_lh[0] = copy_Exp( w->lh );
	  tmp2->numeric_preconds_rh[0] = copy_Exp( w->rh );
	  tmp2->num_numeric_preconds = 1;
	}
      }
      tmp2->effects = instantiate_Effect( o->effects );
      tmp2->next = ghard_mixed_operators;
      ghard_mixed_operators = tmp2;
      gnum_hard_mixed_operators++;
    }
    break;
  case AND:
    tmp2 = new_MixedOperator( o );
    for ( i = 0; i < o->num_vars; i++ ) {
      tmp2->inst_table[i] = linst_table[i];
    }
    m = 0;
    mn = 0;
    for ( w = tmp1->sons; w; w = w->next ) {
      if ( w->connective == ATOM ) m++;
      if ( w->connective == COMP ) mn++;
    }
    tmp2->preconds = ( Fact * ) calloc( m, sizeof( Fact ) );
    tmp2->numeric_preconds_comp = ( Comparator * ) calloc( mn, sizeof( Comparator ) );
    tmp2->numeric_preconds_lh = ( ExpNode_pointer * ) calloc( mn, sizeof( ExpNode_pointer ) );
    tmp2->numeric_preconds_rh = ( ExpNode_pointer * ) calloc( mn, sizeof( ExpNode_pointer ) );
    tmp2->num_preconds = m;
    tmp2->num_numeric_preconds = mn;
    m = 0; mn = 0;
    for ( w = tmp1->sons; w; w = w->next ) {
      if ( w->connective == ATOM ) {
	tmp2->preconds[m].predicate = w->fact->predicate;
	for ( i = 0; i < garity[w->fact->predicate]; i++ ) {
	  tmp2->preconds[m].args[i] = w->fact->args[i];
	}
	m++;
      }
      if ( w->connective == COMP ) {
	tmp2->numeric_preconds_comp[mn] = w->comp;
	tmp2->numeric_preconds_lh[mn] = copy_Exp( w->lh );
	tmp2->numeric_preconds_rh[mn] = copy_Exp( w->rh );
	mn++;
      }
    }
    tmp2->effects = instantiate_Effect( o->effects );
    tmp2->next = ghard_mixed_operators;
    ghard_mixed_operators = tmp2;
    gnum_hard_mixed_operators++;
    break;
  case ATOM:
    tmp2 = new_MixedOperator( o );
    for ( i = 0; i < o->num_vars; i++ ) {
      tmp2->inst_table[i] = linst_table[i];
    }
    tmp2->preconds = ( Fact * ) calloc( 1, sizeof( Fact ) );
    tmp2->num_preconds = 1;
    tmp2->preconds[0].predicate = tmp1->fact->predicate;
    for ( i = 0; i < garity[tmp1->fact->predicate]; i++ ) {
      tmp2->preconds[0].args[i] = tmp1->fact->args[i];
    }
    tmp2->effects = instantiate_Effect( o->effects );
    tmp2->next = ghard_mixed_operators;
    ghard_mixed_operators = tmp2;
    gnum_hard_mixed_operators++;
    break;
  case COMP:
    tmp2 = new_MixedOperator( o );
    for ( i = 0; i < o->num_vars; i++ ) {
      tmp2->inst_table[i] = linst_table[i];
    }
    tmp2->numeric_preconds_comp = ( Comparator * ) calloc( 1, sizeof( Comparator ) );
    tmp2->numeric_preconds_lh = ( ExpNode_pointer * ) calloc( 1, sizeof( ExpNode_pointer ) );
    tmp2->numeric_preconds_rh = ( ExpNode_pointer * ) calloc( 1, sizeof( ExpNode_pointer ) );
    tmp2->numeric_preconds_comp[0] = tmp1->comp;
    tmp2->numeric_preconds_lh[0] = copy_Exp( tmp1->lh );
    tmp2->numeric_preconds_rh[0] = copy_Exp( tmp1->rh );
    tmp2->num_numeric_preconds = 1;
    tmp2->effects = instantiate_Effect( o->effects );
    tmp2->next = ghard_mixed_operators;
    ghard_mixed_operators = tmp2;
    gnum_hard_mixed_operators++;    
    break;
  case TRU:
    tmp2 = new_MixedOperator( o );
    for ( i = 0; i < o->num_vars; i++ ) {
      tmp2->inst_table[i] = linst_table[i];
    }
    tmp2->effects = instantiate_Effect( o->effects );
    tmp2->next = ghard_mixed_operators;
    ghard_mixed_operators = tmp2;
    gnum_hard_mixed_operators++;
    break;
  default:
    printf("\n\nillegal connective %d in parsing DNF precond.\n\n",
	   tmp1->connective);
    exit( 1 );
  }

  free_WffNode( tmp1 );

}



Effect *instantiate_Effect( Effect *e )

{

  Effect *res = NULL, *tmp, *i;
  Literal *tt, *l;
  NumericEffect *ne, *ttt;
  int j;

  for ( i = e; i; i = i->next ) {
    tmp = new_Effect();

    for ( j = 0; j < i->num_vars; j++ ) {
      tmp->var_types[j] = i->var_types[j];
    }
    tmp->num_vars = i->num_vars;

    tmp->conditions = instantiate_wff( i->conditions );

    if ( tmp->conditions->connective == FAL ) {
      free_partial_Effect( tmp );
      continue;
    }

    for ( l = i->effects; l; l = l->next ) {
      tt = new_Literal();
      tt->negated = l->negated;
      tt->fact.predicate = l->fact.predicate;
      for ( j = 0; j < garity[tt->fact.predicate]; j++ ) {
	tt->fact.args[j] = l->fact.args[j];
	if ( tt->fact.args[j] < 0 &&
	     linst_table[DECODE_VAR( tt->fact.args[j] )] != -1 ) {
	  tt->fact.args[j] = linst_table[DECODE_VAR( tt->fact.args[j] )];
	}
      }
      tt->next = tmp->effects;
      if ( tmp->effects ) {
	tmp->effects->prev = tt;
      }
      tmp->effects = tt;
    }

    for ( ne = i->numeric_effects; ne; ne = ne->next ) {
      ttt = new_NumericEffect();
      ttt->neft = ne->neft;
      ttt->fluent.function = ne->fluent.function;
      for ( j = 0; j < gf_arity[ttt->fluent.function]; j++ ) {
	ttt->fluent.args[j] = ne->fluent.args[j];
	if ( ttt->fluent.args[j] < 0 &&
	     linst_table[DECODE_VAR( ttt->fluent.args[j] )] != -1 ) {
	  ttt->fluent.args[j] = linst_table[DECODE_VAR( ttt->fluent.args[j] )];
	}
      }
      ttt->rh = copy_Exp( ne->rh );
      instantiate_exp( &(ttt->rh) );
      ttt->next = tmp->numeric_effects;
      if ( tmp->numeric_effects ) {
	tmp->numeric_effects->prev = ttt;
      }
      tmp->numeric_effects = ttt;
    }

    tmp->next = res;
    if ( res ) {
      res->prev = tmp;
    }
    res = tmp;
  }

  return res;

}



WffNode *instantiate_wff( WffNode *w )

{

  WffNode *res = NULL, *tmp, *i;
  int j, m, h;
  Bool ok, ct;

  switch ( w->connective ) {
  case AND:
    m = 0;
    i = w->sons;
    while ( i ) {
      tmp = instantiate_wff( i );
      if ( tmp->connective == FAL ) {
	free_WffNode( res );
	return tmp;
      }
      if ( tmp->connective == TRU ) {
	free( tmp );
	i = i->next;
	continue;
      }
      tmp->next = res;
      if ( res ) {
	res->prev = tmp;
      }
      res = tmp;
      i = i->next;
      m++;
    }
    if ( m == 0 ) {
      res = new_WffNode( TRU );
      break;
    }
    if ( m == 1 ) {
      break;
    }
    tmp = new_WffNode( AND );
    tmp->sons = res;
    res = tmp;
    break;
  case OR:
    m = 0;
    i = w->sons;
    while ( i ) {
      tmp = instantiate_wff( i );
      if ( tmp->connective == TRU ) {
	free_WffNode( res );
	return tmp;
      }
      if ( tmp->connective == FAL ) {
	free( tmp );
	i = i->next;
	continue;
      }
      tmp->next = res;
      if ( res ) {
	res->prev = tmp;
      }
      res = tmp;
      i = i->next;
      m++;
    }
    if ( m == 0 ) {
      res = new_WffNode( FAL );
      break;
    }
    if ( m == 1 ) {
      break;
    }
    tmp = new_WffNode( OR );
    tmp->sons = res;
    res = tmp;
    break;
  case ATOM:
    res = new_WffNode( ATOM );
    res->fact = new_Fact();
    res->fact->predicate = w->fact->predicate;
    ok = TRUE;
    for ( j = 0; j < garity[res->fact->predicate]; j++ ) {
      h = ( w->fact->args[j] < 0 ) ?
	linst_table[DECODE_VAR( w->fact->args[j] )] : w->fact->args[j];
      if ( h < 0 ) {
	ok = FALSE;
	res->fact->args[j] = w->fact->args[j];
      } else {
	res->fact->args[j] = h;
      }
    }
    if ( !ok ) {/* contains ef params */
      break;
    }
    if ( !full_possibly_negative( res->fact ) ) {
      free( res->fact );
      res->fact = NULL;
      res->connective = TRU;
      break;
    }
    if ( !full_possibly_positive( res->fact ) ) {
      free( res->fact );
      res->fact = NULL;
      res->connective = FAL;
      break;
    }
    break;
  case COMP:
    res = new_WffNode( COMP );
    res->comp = w->comp;
    res->lh = copy_Exp( w->lh );
    res->rh = copy_Exp( w->rh );
    instantiate_exp( &(res->lh) );
    instantiate_exp( &(res->rh) );
    if ( res->lh->connective != NUMBER ||
	 res->rh->connective != NUMBER ) {
      /* logical simplification only possible if both parts are numbers
       */
      break;
    }
    ct = number_comparison_holds( res->comp, res->lh->value, res->rh->value );
    if ( ct ) {
      res->connective = TRU;
      free_ExpNode( res->lh );
      res->lh = NULL;
      free_ExpNode( res->rh );
      res->rh = NULL;
      res->comp = -1;
    } else {
      res->connective = FAL;
      free_ExpNode( res->lh );
      res->lh = NULL;
      free_ExpNode( res->rh );
      res->rh = NULL;
      res->comp = -1;
    }
    break;
  case TRU:
  case FAL:
    res = new_WffNode( w->connective );
    break;
  default:
    printf("\n\nillegal connective %d in instantiate formula\n\n",
	   w->connective);
    exit( 1 );
  }

  return res;

}



void instantiate_exp( ExpNode **n )

{

  int j, f, k, h;
  Bool ok;

  switch ( (*n)->connective ) {
  case AD:
    instantiate_exp( &((*n)->leftson) );
    instantiate_exp( &((*n)->rightson) );
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
    instantiate_exp( &((*n)->leftson) );
    instantiate_exp( &((*n)->rightson) );
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
    instantiate_exp( &((*n)->leftson) );
    instantiate_exp( &((*n)->rightson) );
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
    instantiate_exp( &((*n)->leftson) );
    instantiate_exp( &((*n)->rightson) );
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
    instantiate_exp( &((*n)->son) );
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
	linst_table[DECODE_VAR( (*n)->fluent->args[j] )] : (*n)->fluent->args[j];
      if ( h < 0 ) {
	ok = FALSE;
      } else {
	(*n)->fluent->args[j] = h;
      }
    }
    if ( !ok ) {
      break;
    }
    /* we handle only the case where the fluent is fully instantiated,
     * static, and in the initial state.
     */
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
    printf("\n\ninst exp: wrong specifier %d",
	   (*n)->connective);
    exit( 1 );
  }

}



Bool full_possibly_positive( Fact *f )

{

  int adr;

  if ( gis_added[f->predicate] ) {
    return TRUE;
  }

  adr = instantiated_fact_adress( f );

  if ( lini[f->predicate][adr] > 0 ) {
    return TRUE;
  } else {
    return FALSE;
  }

}



Bool full_possibly_negative( Fact *f )

{

  int adr;

  if ( gis_deleted[f->predicate] ) {
    return TRUE;
  }

  adr = instantiated_fact_adress( f );

  if ( lini[f->predicate][adr] > 0 ) {
    return FALSE;
  } else {
    return TRUE;
  }

}



int instantiated_fact_adress( Fact *f )

{

  int r = 0, b = 1, i;

  for ( i = 0; i < garity[f->predicate]; i++ ) {
    r += b * f->args[i];
    b *= gnum_constants;
  }

  return r;

}














/*********************************************************
 * CODE THAT MULTIPLIES EFFECT PARAMS --> PSEUDO ACTIONS *
 *********************************************************/















void multiply_hard_effect_parameters( void )

{

  MixedOperator *o;
  PseudoAction *tmp;
  int i;
  Effect *e;

  ghard_templates = ( PseudoAction_pointer * ) 
    calloc( gnum_hard_mixed_operators, sizeof ( PseudoAction_pointer ) );
  gnum_hard_templates = 0;

  for ( o = ghard_mixed_operators; o; o = o->next ) {
    tmp = new_PseudoAction( o );

    for ( i = 0; i < tmp->operator->num_vars; i++ ) {
      linst_table[i] = tmp->inst_table[i];
    }

    for ( e = o->effects; e; e = e->next ) {
      create_hard_pseudo_effects( tmp, e, 0 );
    }

    ghard_templates[gnum_hard_templates++] = tmp;
  }
}



void create_hard_pseudo_effects( PseudoAction *a, Effect *e, int curr_var )

{

  int par, t, i, m, mn;
  WffNode *tmp1, *w, *ww;
  PseudoActionEffect *tmp2;

  if ( curr_var < e->num_vars ) {
    par = a->operator->num_vars + curr_var;

    t = e->var_types[curr_var];
    for ( i = 0; i < gtype_size[t]; i++ ) {
      linst_table[par] = gtype_consts[t][i];

      create_hard_pseudo_effects( a, e, curr_var + 1 );

      linst_table[par] = -1;
    }
    return;
  }

  tmp1 = instantiate_wff( e->conditions );

  if ( tmp1->connective == FAL ) {
    free_WffNode( tmp1 );
    return;
  }

  dnf( &tmp1 );
  cleanup_wff( &tmp1 );

  /* only debugging, REMOVE LATER
   */
  if ( is_dnf( tmp1 ) == -1 ) {
    printf("\n\nILLEGAL DNF %s AFTER INSTANTIATION\n\n", a->operator->name);
    print_Wff( tmp1, 0 );
    exit( 1 );
  }

  switch ( tmp1->connective ) {
  case OR:
    for ( w = tmp1->sons; w; w = w->next ) {
      tmp2 = new_PseudoActionEffect();
      if ( w->connective == AND ) {
	m = 0;
	mn = 0;
	for ( ww = w->sons; ww; ww = ww->next ) {
	  if ( ww->connective == ATOM ) m++;
	  if ( ww->connective == COMP ) mn++;
	}
	tmp2->conditions = ( Fact * ) calloc( m, sizeof( Fact ) );
	tmp2->numeric_conditions_comp = ( Comparator * ) calloc( mn, sizeof( Comparator ) );
	tmp2->numeric_conditions_lh = ( ExpNode_pointer * ) calloc( mn, sizeof( ExpNode_pointer ) );
	tmp2->numeric_conditions_rh = ( ExpNode_pointer * ) calloc( mn, sizeof( ExpNode_pointer ) );
	tmp2->num_conditions = m;
	tmp2->num_numeric_conditions = mn;
	m = 0; mn = 0;
	for ( ww = w->sons; ww; ww = ww->next ) {
	  if ( ww->connective == ATOM ) {
	    tmp2->conditions[m].predicate = ww->fact->predicate;
	    for ( i = 0; i < garity[ww->fact->predicate]; i++ ) {
	      tmp2->conditions[m].args[i] = ww->fact->args[i];
	    }
	    m++;
	  }
	  if ( ww->connective == COMP ) {
	    tmp2->numeric_conditions_comp[mn] = ww->comp;
	    tmp2->numeric_conditions_lh[mn] = copy_Exp( ww->lh );
	    tmp2->numeric_conditions_rh[mn] = copy_Exp( ww->rh );
	    mn++;
	  }
	}
      } else {
	if ( w->connective == ATOM ) {
	  tmp2->conditions = ( Fact * ) calloc( 1, sizeof( Fact ) );
	  tmp2->num_conditions = 1;
	  tmp2->conditions[0].predicate = w->fact->predicate;
	  for ( i = 0; i < garity[w->fact->predicate]; i++ ) {
	    tmp2->conditions[0].args[i] = w->fact->args[i];
	  }
	}
 	if ( w->connective == COMP ) {
	  tmp2->numeric_conditions_comp = ( Comparator * ) calloc( 1, sizeof( Comparator ) );
	  tmp2->numeric_conditions_lh = ( ExpNode_pointer * ) calloc( 1, sizeof( ExpNode_pointer ) );
	  tmp2->numeric_conditions_rh = ( ExpNode_pointer * ) calloc( 1, sizeof( ExpNode_pointer ) );
	  tmp2->numeric_conditions_comp[0] = w->comp;
	  tmp2->numeric_conditions_lh[0] = copy_Exp( w->lh );
	  tmp2->numeric_conditions_rh[0] = copy_Exp( w->rh );
	  tmp2->num_numeric_conditions = 1;
	}
      }
      make_instantiate_literals( tmp2, e->effects );
      make_instantiate_numeric_effects( tmp2, e->numeric_effects );
      tmp2->next = a->effects;
      a->effects = tmp2;
      a->num_effects++;
    }
    break;
  case AND:
    tmp2 = new_PseudoActionEffect();
    m = 0;
    mn = 0;
    for ( w = tmp1->sons; w; w = w->next ) {
      if ( w->connective == ATOM ) m++;
      if ( w->connective == COMP ) mn++;
    }
    tmp2->conditions = ( Fact * ) calloc( m, sizeof( Fact ) );
    tmp2->numeric_conditions_comp = ( Comparator * ) calloc( mn, sizeof( Comparator ) );
    tmp2->numeric_conditions_lh = ( ExpNode_pointer * ) calloc( mn, sizeof( ExpNode_pointer ) );
    tmp2->numeric_conditions_rh = ( ExpNode_pointer * ) calloc( mn, sizeof( ExpNode_pointer ) );
    tmp2->num_conditions = m;
    tmp2->num_numeric_conditions = mn;
    m = 0; mn = 0;
    for ( w = tmp1->sons; w; w = w->next ) {
      if ( w->connective == ATOM ) {
	tmp2->conditions[m].predicate = w->fact->predicate;
	for ( i = 0; i < garity[w->fact->predicate]; i++ ) {
	  tmp2->conditions[m].args[i] = w->fact->args[i];
	}
	m++;
      }
      if ( w->connective == COMP ) {
	tmp2->numeric_conditions_comp[mn] = w->comp;
	tmp2->numeric_conditions_lh[mn] = copy_Exp( w->lh );
	tmp2->numeric_conditions_rh[mn] = copy_Exp( w->rh );
	mn++;
      }
    }
    make_instantiate_literals( tmp2, e->effects );
    make_instantiate_numeric_effects( tmp2, e->numeric_effects );
    tmp2->next = a->effects;
    a->effects = tmp2;
    a->num_effects++;
    break;
  case ATOM:
    tmp2 = new_PseudoActionEffect();
    tmp2->conditions = ( Fact * ) calloc( 1, sizeof( Fact ) );
    tmp2->num_conditions = 1;
    tmp2->conditions[0].predicate = tmp1->fact->predicate;
    for ( i = 0; i < garity[tmp1->fact->predicate]; i++ ) {
      tmp2->conditions[0].args[i] = tmp1->fact->args[i];
    }
    make_instantiate_literals( tmp2, e->effects );
    make_instantiate_numeric_effects( tmp2, e->numeric_effects );
    tmp2->next = a->effects;
    a->effects = tmp2;
    a->num_effects++;
    break;
  case COMP:
    tmp2 = new_PseudoActionEffect();
    tmp2->numeric_conditions_comp = ( Comparator * ) calloc( 1, sizeof( Comparator ) );
    tmp2->numeric_conditions_lh = ( ExpNode_pointer * ) calloc( 1, sizeof( ExpNode_pointer ) );
    tmp2->numeric_conditions_rh = ( ExpNode_pointer * ) calloc( 1, sizeof( ExpNode_pointer ) );
    tmp2->numeric_conditions_comp[0] = tmp1->comp;
    tmp2->numeric_conditions_lh[0] = copy_Exp( tmp1->lh );
    tmp2->numeric_conditions_rh[0] = copy_Exp( tmp1->rh );
    tmp2->num_numeric_conditions = 1;
    make_instantiate_literals( tmp2, e->effects );
    make_instantiate_numeric_effects( tmp2, e->numeric_effects );
    tmp2->next = a->effects;
    a->effects = tmp2;
    a->num_effects++;
    break;
  case TRU:
    tmp2 = new_PseudoActionEffect();
    make_instantiate_literals( tmp2, e->effects );
    make_instantiate_numeric_effects( tmp2, e->numeric_effects );
    tmp2->next = a->effects;
    a->effects = tmp2;
    a->num_effects++;
    break;
  default:
    printf("\n\nillegal connective %d in parsing DNF condition.\n\n",
	   tmp1->connective);
    exit( 1 );
  }

  free_WffNode( tmp1 );

}
 


void make_instantiate_literals( PseudoActionEffect *e, Literal *ll )

{

  int ma = 0, md = 0, i;
  Literal *l;

  for ( l = ll; l; l = l->next ) {
    if ( l->negated ) {
      md++;
    } else {
      ma++;
    }
  }

  e->adds = ( Fact * ) calloc( ma, sizeof( Fact ) );
  e->dels = ( Fact * ) calloc( md, sizeof( Fact ) );

  for ( l = ll; l; l = l->next ) {
    if ( l->negated ) {
      e->dels[e->num_dels].predicate = l->fact.predicate;
      for ( i = 0; i < garity[l->fact.predicate]; i++ ) {
	e->dels[e->num_dels].args[i] = ( l->fact.args[i] < 0 ) ?
	  linst_table[DECODE_VAR( l->fact.args[i] )] : l->fact.args[i];
      }
      e->num_dels++;
    } else {
      e->adds[e->num_adds].predicate = l->fact.predicate;
      for ( i = 0; i < garity[l->fact.predicate]; i++ ) {
	e->adds[e->num_adds].args[i] = ( l->fact.args[i] < 0 ) ?
	  linst_table[DECODE_VAR( l->fact.args[i] )] : l->fact.args[i];
      }
      e->num_adds++;
    }
  }

}
 


void make_instantiate_numeric_effects( PseudoActionEffect *e, NumericEffect *ne )

{

  int m = 0, i;
  NumericEffect *n;

  for ( n = ne; n; n = n->next ) m++;

  e->numeric_effects_neft = ( NumericEffectType * ) calloc( m, sizeof( NumericEffectType ) );
  e->numeric_effects_fluent = ( Fluent * ) calloc( m, sizeof( Fluent ) );
  e->numeric_effects_rh = ( ExpNode_pointer * ) calloc( m, sizeof( ExpNode_pointer ) );
  e->num_numeric_effects = m;

  m = 0;
  for ( n = ne; n; n = n->next ) {
    e->numeric_effects_neft[m] = n->neft;
    e->numeric_effects_fluent[m].function = n->fluent.function;
    for ( i = 0; i < gf_arity[n->fluent.function]; i++ ) {
      e->numeric_effects_fluent[m].args[i] = ( n->fluent.args[i] < 0 ) ?
	linst_table[DECODE_VAR( n->fluent.args[i] )] : n->fluent.args[i];
    }
    e->numeric_effects_rh[m] = copy_Exp( n->rh );
    instantiate_exp( &(e->numeric_effects_rh[m]) );
    m++;
  }

}
