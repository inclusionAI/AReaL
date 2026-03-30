
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
 * File: parse.c
 * Description: Functions for the pddl parser
 *
 * Author: Joerg Hoffmann 2000
 *
 *********************************************************************/ 






#include "ff.h"

#include "memory.h"
#include "output.h"

#include "parse.h"











/* simple parse helpers
 */







char *copy_Token( char *s )

{

  char *d = new_Token( strlen( s ) + 1 );
  strcpy(d, s);

  return d;

}



TokenList *copy_TokenList( TokenList *source )

{

  TokenList *temp;

  if ( !source ) {
    temp = NULL;
  } else {
    temp = new_TokenList();
    if ( source->item ) {
      temp->item = new_Token( strlen( source->item ) + 1 );
      strcpy( temp->item, source->item );
    }
    temp->next = copy_TokenList( source->next );
  }

  return temp;

}



void strupcase( char *from )

{

  char tmp;

  tmp = *from;
  while ('\0' != tmp) {
    *from = (char) toupper((int) tmp);
    tmp = *++from;
  }

}



char *rmdash( char *s )

{

  s++;

  for( ; (*s == ' ') || (*s == '\t'); s++ );

  return s;

}










/* typed-list-of   preprocessing
 */







Token ltype_names[MAX_TYPES];
int lnum_types;


int leither_ty[MAX_TYPES][MAX_TYPES];
int lnum_either_ty[MAX_TYPES];





void build_orig_constant_list( void )

{

  char *tmp = NULL;
  TypedList *tyl;
  TypedListList *tyll;
  TokenList *tl, *p_tl, *tmp_tl;
  PlOperator *po;

  int i, j, k, n, std;

  Bool m[MAX_TYPES][MAX_TYPES];

  FactList *fl, *p_fl;

  lnum_types = 0;
  for ( tyl = gparse_types; tyl; tyl = tyl->next ) {
    if ( get_type( tyl->name ) == -1 ) {
      ltype_names[lnum_types++] = copy_Token( tyl->name );
    }
    if ( tyl->type->next ) {
      tmp = new_Token( MAX_LENGTH );
      strcpy( tmp, EITHER_STR );
      for ( tl = tyl->type; tl; tl = tl->next ) {
	strcat( tmp, CONNECTOR );
	strcat( tmp, tl->item );
      }
    } else {
      tmp = copy_Token( tyl->type->item );
    }
    if ( (n = get_type( tmp )) == -1 ) {
      tyl->n = lnum_types;
      ltype_names[lnum_types++] = copy_Token( tmp );
    } else {
      tyl->n = n;
    }
    free( tmp );
    tmp = NULL;
  }
     
  for ( tyl = gparse_constants; tyl; tyl = tyl->next ) {
    if ( tyl->type->next ) {
      tmp = new_Token( MAX_LENGTH );
      strcpy( tmp, EITHER_STR );
      for ( tl = tyl->type; tl; tl = tl->next ) {
	strcat( tmp, CONNECTOR );
	strcat( tmp, tl->item );
      }
    } else {
      tmp = copy_Token( tyl->type->item );
    }
    if ( (n = get_type( tmp )) == -1 ) {
      tyl->n = lnum_types;
      ltype_names[lnum_types++] = copy_Token( tmp );
    } else {
      tyl->n = n;
    }
    free( tmp );
    tmp = NULL;
  }
  
  for ( tyl = gparse_objects; tyl; tyl = tyl->next ) {
    if ( tyl->type->next ) {
      tmp = new_Token( MAX_LENGTH );
      strcpy( tmp, EITHER_STR );
      for ( tl = tyl->type; tl; tl = tl->next ) {
	strcat( tmp, CONNECTOR );
	strcat( tmp, tl->item );
      }
    } else {
      tmp = copy_Token( tyl->type->item );
    }
    if ( (n = get_type( tmp )) == -1 ) {
      tyl->n = lnum_types;
      ltype_names[lnum_types++] = copy_Token( tmp );
    } else {
      tyl->n = n;
    }
    free( tmp );
    tmp = NULL;
  }

  for ( tyll = gparse_predicates; tyll; tyll = tyll->next ) {
    for ( tyl = tyll->args; tyl; tyl = tyl->next ) {
      if ( tyl->type->next ) {
	tmp = new_Token( MAX_LENGTH );
	strcpy( tmp, EITHER_STR );
	for ( tl = tyl->type; tl; tl = tl->next ) {
	  strcat( tmp, CONNECTOR );
	  strcat( tmp, tl->item );
	}
      } else {
	tmp = copy_Token( tyl->type->item );
      }
      if ( (n = get_type( tmp )) == -1 ) {
	tyl->n = lnum_types;
	ltype_names[lnum_types++] = copy_Token( tmp );
      } else {
	tyl->n = n;
      }
      free( tmp );
      tmp = NULL;
    }
  }
    
  for ( tyll = gparse_functions; tyll; tyll = tyll->next ) {
    for ( tyl = tyll->args; tyl; tyl = tyl->next ) {
      if ( tyl->type->next ) {
 	tmp = new_Token( MAX_LENGTH );
 	strcpy( tmp, EITHER_STR );
 	for ( tl = tyl->type; tl; tl = tl->next ) {
 	  strcat( tmp, CONNECTOR );
 	  strcat( tmp, tl->item );
 	}
      } else {
 	tmp = copy_Token( tyl->type->item );
      }
      if ( (n = get_type( tmp )) == -1 ) {
 	tyl->n = lnum_types;
 	ltype_names[lnum_types++] = copy_Token( tmp );
      } else {
 	tyl->n = n;
      }
      free( tmp );
      tmp = NULL;
    }
  }
    
  collect_type_names_in_pl( gorig_goal_facts );

  for ( po = gloaded_ops; po; po = po->next ) {
    collect_type_names_in_pl( po->preconds );
    collect_type_names_in_pl( po->effects );
    for ( tyl = po->parse_params; tyl; tyl = tyl->next ) {
      if ( tyl->type->next ) {
	tmp = new_Token( MAX_LENGTH );
	strcpy( tmp, EITHER_STR );
	for ( tl = tyl->type; tl; tl = tl->next ) {
	  strcat( tmp, CONNECTOR );
	  strcat( tmp, tl->item );
	}
      } else {
	tmp = copy_Token( tyl->type->item );
      }
      if ( (n = get_type( tmp )) == -1 ) {
	tyl->n = lnum_types;
	ltype_names[lnum_types++] = copy_Token( tmp );
      } else {
	tyl->n = n;
      }
      free( tmp );
      tmp = NULL;
    }
  }


  /* now get the numbers of all composed either types
   */
  for ( i = 0; i < lnum_types; i++ ) {
    lnum_either_ty[i] = 0;
  }
  for ( tyl = gparse_types; tyl; tyl = tyl->next ) {
    make_either_ty( tyl );
  }
  for ( tyl = gparse_constants; tyl; tyl = tyl->next ) {
    make_either_ty( tyl );
  }
  for ( tyl = gparse_objects; tyl; tyl = tyl->next ) {
    make_either_ty( tyl );
  }
  for ( tyll = gparse_predicates; tyll; tyll = tyll->next ) {
    for ( tyl = tyll->args; tyl; tyl = tyl->next ) {
      make_either_ty( tyl );
    }
  }
  for ( tyll = gparse_functions; tyll; tyll = tyll->next ) {
    for ( tyl = tyll->args; tyl; tyl = tyl->next ) {
      make_either_ty( tyl );
    }
  }
  make_either_ty_in_pl( gorig_goal_facts );
  for ( po = gloaded_ops; po; po = po->next ) {
    make_either_ty_in_pl( po->preconds );
    make_either_ty_in_pl( po->effects );
    for ( tyl = po->parse_params; tyl; tyl = tyl->next ) {
      make_either_ty( tyl );
    }
  }


  /* now, compute the transitive closure of all type inclusions.
   * first initialize the matrix.
   */
  for ( i = 0; i < lnum_types; i++ ) {
    for ( j = 0; j < lnum_types; j++ ) {
      m[i][j] = ( i == j ? TRUE : FALSE );
    }
  }
  std = -1;
  for ( i = 0; i < lnum_types; i++ ) {
    if ( strcmp( ltype_names[i], STANDARD_TYPE ) == SAME ) {
      std = i;
      break;
    }
  }
  for ( i = 0; i < lnum_types; i++ ) {
    m[i][std] = TRUE;/* all types are subtypes of OBJECT */
  }
  for ( tyl = gparse_types; tyl; tyl = tyl->next ) {
    /* all inclusions as are defined in domain file
     */
    m[get_type( tyl->name )][tyl->n] = TRUE;
  }
  /* compute transitive closure on inclusions matrix
   */
  for ( j = 0; j < lnum_types; j++ ) {
    for ( i = 0; i < lnum_types; i++ ) {
      if ( m[i][j] ) {
	for ( k = 0; k < lnum_types; k++ ) {
	  if ( m[j][k] ) {
	    m[i][k] = TRUE;
	  }
	}
      }
    }
  }
  /* union types are subsets of all those types that contain all
   * their components, and 
   * all component types are subsets of the either type !
   */
  for ( i = 0; i < lnum_types; i++ ) {
    if ( lnum_either_ty[i] < 2 ) continue;
    for ( j = 0; j < lnum_types; j++ ) {
      if ( j == i ) continue;
      /* get supertypes of all component types
       */
      for ( k = 0; k < lnum_either_ty[i]; k++ ) {
	if ( !m[leither_ty[i][k]][j] ) break;
      }
      if ( k < lnum_either_ty[i] ) continue;
      m[i][j] = TRUE;
      /* make components subtypes of either type
       */
      for ( k = 0; k < lnum_either_ty[i]; k++ ) {
	m[leither_ty[i][k]][i] = TRUE;
      }
    }
  }
  /* and again, compute transitive closure on inclusions matrix.
   * I guess, this won't change anything (?), but it also won't need
   * any remarkable computation time, so why should one think about it ?
   */
  for ( j = 0; j < lnum_types; j++ ) {
    for ( i = 0; i < lnum_types; i++ ) {
      if ( m[i][j] ) {
	for ( k = 0; k < lnum_types; k++ ) {
	  if ( m[j][k] ) {
	    m[i][k] = TRUE;
	  }
	}
      }
    }
  }
  

  /* now build FactList of ALL  constant -> type   pairs.
   * for each constant / object, let it appear separately
   * for each type it is a member of; compute type
   * membership based on propagating constants / objects
   * through inclusions matrix.
   *
   * this might make the same pair appear doubly, if an object
   * is declared in type T as well as in some supertype T'.
   * such cases will be filtered out in string collection.
   */
  for ( tyl = gparse_constants; tyl; tyl = tyl->next ) {
    fl = new_FactList();
    fl->item = new_TokenList();
    fl->item->next = new_TokenList();
    fl->item->item = copy_Token( tyl->name );
    if ( tyl->type->next ) {
      fl->item->next->item = new_Token( MAX_LENGTH );
      strcpy( fl->item->next->item, EITHER_STR );
      for ( tl = tyl->type; tl; tl = tl->next ) {
	strcat( fl->item->next->item, CONNECTOR );
	strcat( fl->item->next->item, tl->item );
      }
    } else {
      fl->item->next->item = copy_Token( tyl->type->item );
    }
    fl->next = gorig_constant_list;
    gorig_constant_list = fl;
    /* now add constant to all supertypes
     */
    n = get_type( fl->item->next->item );
    for ( i = 0; i < lnum_types; i++ ) {
      if ( i == n ||
	   !m[n][i] ) continue;
      fl = new_FactList();
      fl->item = new_TokenList();
      fl->item->next = new_TokenList();
      fl->item->item = copy_Token( tyl->name );
      fl->item->next->item = copy_Token( ltype_names[i] );
      fl->next = gorig_constant_list;
      gorig_constant_list = fl;
    }
  }
  for ( tyl = gparse_objects; tyl; tyl = tyl->next ) {
    fl = new_FactList();
    fl->item = new_TokenList();
    fl->item->next = new_TokenList();
    fl->item->item = copy_Token( tyl->name );
    if ( tyl->type->next ) {
      fl->item->next->item = new_Token( MAX_LENGTH );
      strcpy( fl->item->next->item, EITHER_STR );
      for ( tl = tyl->type; tl; tl = tl->next ) {
	strcat( fl->item->next->item, CONNECTOR );
	strcat( fl->item->next->item, tl->item );
      }
    } else {
      fl->item->next->item = copy_Token( tyl->type->item );
    }
    fl->next = gorig_constant_list;
    gorig_constant_list = fl;
    /* now add constant to all supertypes
     */
    n = get_type( fl->item->next->item );
    for ( i = 0; i < lnum_types; i++ ) {
      if ( i == n ||
	   !m[n][i] ) continue;
      fl = new_FactList();
      fl->item = new_TokenList();
      fl->item->next = new_TokenList();
      fl->item->item = copy_Token( tyl->name );
      fl->item->next->item = copy_Token( ltype_names[i] );
      fl->next = gorig_constant_list;
      gorig_constant_list = fl;
    }
  }


  /* now, normalize all typed-list-of  s in domain and problem def,
   * i.e., in all PlNode quantifiers and in op parameters
   *
   * at the same time, remove typed-listof structures in these defs
   */
  normalize_tyl_in_pl( &gorig_goal_facts );
  for ( po = gloaded_ops; po; po = po->next ) {
    normalize_tyl_in_pl( &po->preconds );
    normalize_tyl_in_pl( &po->effects );
    /* be careful to maintain parameter ordering !
     */
    if ( !po->parse_params ) {
      continue;/* no params at all */
    }
    fl = new_FactList();
    fl->item = new_TokenList();
    fl->item->next = new_TokenList();
    fl->item->item = copy_Token( po->parse_params->name );
    if ( po->parse_params->type->next ) {
      fl->item->next->item = new_Token( MAX_LENGTH );
      strcpy( fl->item->next->item, EITHER_STR );
      for ( tl = po->parse_params->type; tl; tl = tl->next ) {
	strcat( fl->item->next->item, CONNECTOR );
	strcat( fl->item->next->item, tl->item );
      }
    } else {
      fl->item->next->item = copy_Token( po->parse_params->type->item );
    }
    po->params = fl;
    p_fl = fl;
    for ( tyl = po->parse_params->next; tyl; tyl = tyl->next ) {
      fl = new_FactList();
      fl->item = new_TokenList();
      fl->item->next = new_TokenList();
      fl->item->item = copy_Token( tyl->name );
      if ( tyl->type->next ) {
	fl->item->next->item = new_Token( MAX_LENGTH );
	strcpy( fl->item->next->item, EITHER_STR );
	for ( tl = tyl->type; tl; tl = tl->next ) {
	  strcat( fl->item->next->item, CONNECTOR );
	  strcat( fl->item->next->item, tl->item );
	}
      } else {
	fl->item->next->item = copy_Token( tyl->type->item );
      }
      p_fl->next = fl;
      p_fl = fl;
    }
    free_TypedList( po->parse_params );
    po->parse_params = NULL;
  }


  /* finally, build  gpredicates_and_types  by chaining predicate names 
   * together with the names of their args' types.
   */
  for ( tyll = gparse_predicates; tyll; tyll = tyll->next ) {
    fl = new_FactList();
    fl->item = new_TokenList();
    fl->item->item = copy_Token( tyll->predicate );
    fl->next = gpredicates_and_types;
    gpredicates_and_types = fl;
    if ( !tyll->args ) continue;
    /* add arg types; MAINTAIN ORDERING !
     */
    fl->item->next = new_TokenList();
    if ( tyll->args->type->next ) {
      fl->item->next->item = new_Token( MAX_LENGTH );
      strcpy( fl->item->next->item, EITHER_STR );
      for ( tl = tyll->args->type; tl; tl = tl->next ) {
	strcat( fl->item->next->item, CONNECTOR );
	strcat( fl->item->next->item, tl->item );
      }
    } else {
      fl->item->next->item = copy_Token( tyll->args->type->item );
    }
    p_tl = fl->item->next;
    for ( tyl = tyll->args->next; tyl; tyl = tyl->next ) {
      tmp_tl = new_TokenList();
      if ( tyl->type->next ) {
	tmp_tl->item = new_Token( MAX_LENGTH );
	strcpy( tmp_tl->item, EITHER_STR );
	for ( tl = tyl->type; tl; tl = tl->next ) {
	  strcat( tmp_tl->item, CONNECTOR );
	  strcat( tmp_tl->item, tl->item );
	}
      } else {
	tmp_tl->item = copy_Token( tyl->type->item );
      }
      p_tl->next = tmp_tl;
      p_tl = tmp_tl;
    }
  }

  for ( tyll = gparse_functions; tyll; tyll = tyll->next ) {
    fl = new_FactList();
    fl->item = new_TokenList();
    fl->item->item = copy_Token( tyll->predicate );
    fl->next = gfunctions_and_types;
    gfunctions_and_types = fl;
    if ( !tyll->args ) continue;
    /* add arg types; MAINTAIN ORDERING !
     */
    fl->item->next = new_TokenList();
    if ( tyll->args->type->next ) {
      fl->item->next->item = new_Token( MAX_LENGTH );
      strcpy( fl->item->next->item, EITHER_STR );
      for ( tl = tyll->args->type; tl; tl = tl->next ) {
	strcat( fl->item->next->item, CONNECTOR );
	strcat( fl->item->next->item, tl->item );
      }
    } else {
      fl->item->next->item = copy_Token( tyll->args->type->item );
    }
    p_tl = fl->item->next;
    for ( tyl = tyll->args->next; tyl; tyl = tyl->next ) {
      tmp_tl = new_TokenList();
      if ( tyl->type->next ) {
	tmp_tl->item = new_Token( MAX_LENGTH );
	strcpy( tmp_tl->item, EITHER_STR );
	for ( tl = tyl->type; tl; tl = tl->next ) {
	  strcat( tmp_tl->item, CONNECTOR );
	  strcat( tmp_tl->item, tl->item );
	}
      } else {
	tmp_tl->item = copy_Token( tyl->type->item );
      }
      p_tl->next = tmp_tl;
      p_tl = tmp_tl;
    }
  }

  /* now get rid of remaining typed-list-of parsing structures
   */
  free_TypedList( gparse_types );
  gparse_types = NULL;
  free_TypedList( gparse_constants );
  gparse_constants = NULL;
  free_TypedList( gparse_objects );
  gparse_objects = NULL;
  free_TypedListList( gparse_predicates );
  gparse_predicates = NULL;
  free_TypedListList( gparse_functions );
  gparse_functions = NULL;

}



void collect_type_names_in_pl( PlNode *n )

{

  PlNode *i;
  TypedList *tyl;
  TokenList *tl;
  char *tmp = NULL;
  int nn;

  if ( !n ) {
    return;
  }

  switch( n->connective ) {
  case ALL:
  case EX:
    for ( tyl = n->parse_vars; tyl; tyl = tyl->next ) {
      if ( tyl->type->next ) {
	tmp = new_Token( MAX_LENGTH );
	strcpy( tmp, EITHER_STR );
	for ( tl = tyl->type; tl; tl = tl->next ) {
	  strcat( tmp, CONNECTOR );
	  strcat( tmp, tl->item );
	}
      } else {
	tmp = copy_Token( tyl->type->item );
      }
      if ( (nn = get_type( tmp )) == -1 ) {
	tyl->n = lnum_types;
	ltype_names[lnum_types++] = copy_Token( tmp );
      } else {
	tyl->n = nn;
      }
      free( tmp );
      tmp = NULL;
    }
    collect_type_names_in_pl( n->sons );
    break;
  case AND:
  case OR:
    for ( i = n->sons; i; i = i->next ) {
      collect_type_names_in_pl( i );
    }
    break;
  case NOT:
    collect_type_names_in_pl( n->sons );
    break;
  case ATOM:
  case TRU:
  case FAL:
    break;
  case WHEN:
    collect_type_names_in_pl( n->sons );
    collect_type_names_in_pl( n->sons->next );
    break;
  default:
    break;
  }

}



int get_type( char *str )

{

  int i;

  for ( i = 0; i < lnum_types; i++ ) {
    if ( strcmp( str, ltype_names[i] ) == SAME ) return i;
  }

  return -1;

}



void make_either_ty( TypedList *tyl )

{

  TokenList *i;

  if ( lnum_either_ty[tyl->n] > 0 ) {
    return;
  }

  for ( i = tyl->type; i; i = i->next ) {
    leither_ty[tyl->n][lnum_either_ty[tyl->n]++] = get_type( i->item );
  }

}



void make_either_ty_in_pl( PlNode *n )

{

  PlNode *i;
  TypedList *tyl;

  if ( !n ) {
    return;
  }

  switch( n->connective ) {
  case ALL:
  case EX:
    for ( tyl = n->parse_vars; tyl; tyl = tyl->next ) {
      make_either_ty( tyl );
    }
    make_either_ty_in_pl( n->sons );
    break;
  case AND:
  case OR:
    for ( i = n->sons; i; i = i->next ) {
      make_either_ty_in_pl( i );
    }
    break;
  case NOT:
    make_either_ty_in_pl( n->sons );
    break;
  case ATOM:
  case TRU:
  case FAL:
    break;
  case WHEN:
    make_either_ty_in_pl( n->sons );
    make_either_ty_in_pl( n->sons->next );
    break;
  default:
    break;
  }

}



void normalize_tyl_in_pl( PlNode **n )

{

  PlNode *i;
  TypedList *tyl;
  PlNode *tmp_pl = NULL, *sons, *p_pl;
  TokenList *tmp_tl, *tl;


  if ( !(*n) ) {
    return;
  }

  switch( (*n)->connective ) {
  case ALL:
  case EX:
    /* we need to make a sequence of quantifiers ( ->sons ...)
     * out of the given sequence of TypedList  elements,
     * with connected type names, var - name in TokenList
     * and KEEPING THE SAME ORDERING !!
     */
    if ( !(*n)->parse_vars ) {
      printf("\n\nquantifier without argument !! check input files.\n\n");
      exit( 1 );
    }
    tmp_tl = new_TokenList();
    tmp_tl->next = new_TokenList();
    tmp_tl->item = copy_Token( (*n)->parse_vars->name );
    if ( (*n)->parse_vars->type->next ) {
      tmp_tl->next->item = new_Token( MAX_LENGTH );
      strcpy( tmp_tl->next->item, EITHER_STR );
      for ( tl = (*n)->parse_vars->type; tl; tl = tl->next ) {
	strcat( tmp_tl->next->item, CONNECTOR );
	strcat( tmp_tl->next->item, tl->item );
      }
    } else {
      tmp_tl->next->item = copy_Token( (*n)->parse_vars->type->item );
    }
    (*n)->atom = tmp_tl;
    /* now add list of sons
     */
    sons = (*n)->sons;
    p_pl = *n;
    for ( tyl = (*n)->parse_vars->next; tyl; tyl = tyl->next ) {
      tmp_tl = new_TokenList();
      tmp_tl->next = new_TokenList();
      tmp_tl->item = copy_Token( tyl->name );
      if ( tyl->type->next ) {
	tmp_tl->next->item = new_Token( MAX_LENGTH );
	strcpy( tmp_tl->next->item, EITHER_STR );
	for ( tl = tyl->type; tl; tl = tl->next ) {
	  strcat( tmp_tl->next->item, CONNECTOR );
	  strcat( tmp_tl->next->item, tl->item );
	}
      } else {
	tmp_tl->next->item = copy_Token( tyl->type->item );
      }
      tmp_pl = new_PlNode( (*n)->connective );
      tmp_pl->atom = tmp_tl;
      p_pl->sons = tmp_pl;
      p_pl = tmp_pl;
    }
    /* remove typed-list-of info
     */
    free_TypedList( (*n)->parse_vars );
    (*n)->parse_vars = NULL;
    /* the last son in list takes over ->sons
     */
    p_pl->sons = sons;
    /* normalize this sons and get out
     */
    normalize_tyl_in_pl( &(p_pl->sons) );
    break;
  case AND:
  case OR:
    for ( i = (*n)->sons; i; i = i->next ) {
      normalize_tyl_in_pl( &i );
    }
    break;
  case NOT:
    normalize_tyl_in_pl( &((*n)->sons) );
    break;
  case ATOM:
  case TRU:
  case FAL:
    break;
  case WHEN:
    normalize_tyl_in_pl( &((*n)->sons) );
    normalize_tyl_in_pl( &((*n)->sons->next) );
    break;
  default:
    break;
  }

}












/* ADL syntax test - and normalization (AND s etc.)
 */












Bool make_adl_domain( void )

{

  PlOperator *i;
  FactList *ff;

  if ( gcmd_line.display_info == 101 ) {
    printf("\noriginal problem parsing is:\n");
    printf("\nobjects:");
    for ( ff = gorig_constant_list; ff; ff = ff->next ) {
      printf("\n%s : %s", ff->item->item, ff->item->next->item);
    }
    printf("\n\ninitial state:\n");
    print_PlNode( gorig_initial_facts, 0 );
    printf("\n\ngoal state:\n");
    print_PlNode( gorig_goal_facts, 0 );
    printf("\n\nops:");
    print_plops( gloaded_ops );
  }

  if ( !make_conjunction_of_atoms( &gorig_initial_facts ) ) {
    printf("\nillegal initial state");
    return FALSE;
  }

  if ( !gorig_goal_facts ) {
    gorig_goal_facts = new_PlNode( TRU );
  }

  if ( !is_wff( gorig_goal_facts ) ) {
    printf("\nillegal goal formula");
    print_PlNode( gorig_goal_facts, 0 );
    return FALSE;
  }

  for ( i = gloaded_ops; i; i = i->next ) {
    if ( !i->preconds ) {
      i->preconds = new_PlNode( TRU );
    }
    if ( !is_wff( i->preconds ) ) {
      printf("\nop %s has illegal precondition", i->name);
      return FALSE;
    }
    if ( !make_effects( &(i->effects) ) ) {
      printf("\nop %s has illegal effects", i->name);
      return FALSE;
    }
  }

  if ( gcmd_line.display_info == 102 ) {
    printf("\nfinal ADL representation is:\n");
    printf("\nobjects:");
    for ( ff = gorig_constant_list; ff; ff = ff->next ) {
      printf("\n%s : %s", ff->item->item, ff->item->next->item);
    }
    printf("\n\ninitial state:\n");
    print_PlNode( gorig_initial_facts, 0 );
    printf("\n\ngoal formula:\n");
    print_PlNode( gorig_goal_facts, 0 );
    printf("\n\nops:");
    print_plops( gloaded_ops );
  }

  return TRUE;
      
}



Bool make_conjunction_of_atoms( PlNode **n )

{

  PlNode *tmp, *i, *p, *m;

  if ( !(*n) ) {
    return TRUE;
  }

  if ( (*n)->connective != AND ) {
    switch ( (*n)->connective ) {
    case ATOM:
      tmp = new_PlNode( ATOM );
      tmp->atom = (*n)->atom;
      (*n)->atom = NULL;
      (*n)->connective = AND;
      (*n)->sons = tmp;
      return TRUE;
    case COMP:
      tmp = new_PlNode( COMP );
      tmp->comp = (*n)->comp;
      tmp->lh = (*n)->lh;
      tmp->rh = (*n)->rh;
      (*n)->lh = NULL;
      (*n)->rh = NULL;
      (*n)->comp = -1;
      (*n)->connective = AND;
      (*n)->sons = tmp;
      return TRUE;
    case NOT:
      free_PlNode( *n );
      (*n) = NULL;
      return TRUE; 
    default:
      return FALSE;
    }
  }

  p = NULL;
  i = (*n)->sons;
  while ( i ) {
    switch ( i->connective ) {
    case ATOM:
      break;
    case COMP:
      break;
    case NOT:
      if ( p ) {
	p->next = i->next;
      } else {
	(*n)->sons = i->next;
      }
      m = i->next;
      i->next = NULL;
      free_PlNode( i );
      i = m;
      break;
    default:
      return FALSE;
    }
    if ( i->connective != NOT ) {
      p = i;
      i = i->next;
    }
  }

  return TRUE;

}



Bool is_wff( PlNode *n )

{

  PlNode *i;

  if ( !n ) {
    return FALSE;
  }

  switch( n->connective ) {
  case ALL:
  case EX:
    if ( !(n->atom) ||
	 !(n->atom->next ) ||
	 n->atom->next->next != NULL ) {
      return FALSE;
    }
    return is_wff( n->sons );
  case AND:
  case OR:
    for ( i = n->sons; i; i = i->next ) {
      if ( !is_wff( i ) ) {
	return FALSE;
      }
    }
    return TRUE;
  case NOT:
    return is_wff( n->sons );
  case ATOM:
    if ( !(n->atom) ||
	 n->sons != NULL ) {
      return FALSE;
    }
    return TRUE;
  case TRU:
  case FAL:
    if ( n->sons != NULL ) {
      return FALSE;
    }
    return TRUE;
  case COMP:
    if ( n->sons != NULL ||
	 n->atom != NULL ||
	 n->lh == NULL ||
	 n->rh == NULL ||
	 n->comp < 0 ) {
      return FALSE;
    }
    return TRUE;
  default:
    return FALSE;
  }

}



Bool make_effects( PlNode **n )

{

  PlNode *tmp, *i, *literals, *j, *k, *next;
  int m = 0;

  if ( (*n)->connective != AND ) {
    if ( !is_eff_literal( *n ) &&
	 (*n)->connective != ALL &&
	 (*n)->connective != WHEN ) {
      return FALSE;
    }
    tmp = new_PlNode( (*n)->connective );
    tmp->atom = (*n)->atom;
    tmp->sons = (*n)->sons;
    tmp->neft = (*n)->neft;
    tmp->lh = (*n)->lh;
    tmp->rh = (*n)->rh;
    (*n)->connective = AND;
    (*n)->sons = tmp;
    (*n)->lh = NULL;
    (*n)->rh = NULL;
    (*n)->neft = -1;
  }

  for ( i = (*n)->sons; i; i = i->next ) {
    if ( is_eff_literal( i ) ) {
      m++;
      continue;
    }
    if ( i->connective == AND ) {
      for ( j = i->sons; j; j = j->next ) {
	if ( !is_eff_literal( j ) ) {
	  return FALSE;
	}
	m++;
      }
      continue;
    }
    if ( i->connective == ALL ) {
      for ( j = i->sons; j && j->connective == ALL; j = j->sons ) {
	if ( !j->atom ||
	     !j->atom->next ||
	     j->atom->next->next != NULL ) {
	  return FALSE;
	}
      }
      if ( !j ) {
	return FALSE;
      }
      if ( is_eff_literal( j ) ) {
	tmp = new_PlNode( AND );
	for ( k = i; k->sons->connective == ALL; k = k->sons );
	k->sons = tmp;
	tmp->sons = j;
	j = tmp;
      }
      if ( j->connective == AND ) {
	for ( k = j->sons; k; k = k->next ) {
	  if ( !is_eff_literal( k ) ) {
	    return FALSE;
	  }
	}
	tmp = new_PlNode( WHEN );
	for ( k = i; k->sons->connective == ALL; k = k->sons );
	k->sons = tmp;
	tmp->sons = new_PlNode( TRU );
	tmp->sons->next = j;
	continue;
      }
      if ( j->connective != WHEN ) {
	return FALSE;
      }
      if ( !(j->sons) ) {
	j->sons = new_PlNode( TRU );
      }
      if ( !is_wff( j->sons ) ) {
	return FALSE;
      }
      if ( !make_conjunction_of_literals( &(j->sons->next) ) ) {
	return FALSE;
      }
      continue;
    }
    if ( i->connective != WHEN ) {
      return FALSE;
    }
    if ( !(i->sons) ) {
      i->sons = new_PlNode( TRU );
    }
    if ( !is_wff( i->sons ) ) {
      return FALSE;
    }
    if ( !make_conjunction_of_literals( &(i->sons->next) ) ) {
      return FALSE;
    }
  }

  if ( m == 0 ) {
    return TRUE;
  }

  tmp = new_PlNode( WHEN );
  tmp->sons = new_PlNode( TRU );
  literals = new_PlNode( AND );
  tmp->sons->next = literals;
  tmp->next = (*n)->sons;
  (*n)->sons = tmp;
  i = (*n)->sons;
  while ( i->next ) {
    if ( is_eff_literal( i->next ) ) {
      next = i->next->next;
      i->next->next = literals->sons;
      literals->sons = i->next;
      i->next = next;
      continue;
    }
    if ( i->next->connective == AND ) {
      next = i->next->next;
      for ( j = i->next->sons; j && j->next; j = j->next );
      if ( j ) {
	j->next = literals->sons;
	literals->sons = i->next->sons;
      }
      i->next = next;
      continue;
    }
    i = i->next;
  }
  return TRUE;

}



Bool is_eff_literal( PlNode *n )

{

  if ( !n ) {
    return FALSE;
  }

  if ( n->connective == NOT ) {
    if ( !n->sons ||
	 n->sons->connective != ATOM ||
	 !n->sons->atom ) {
      return FALSE;
    }
    return TRUE;
  }

  if ( n->connective == ATOM ) {
    if ( !n->atom ) {
      return FALSE;
    }
    return TRUE;
  }

  if ( n->connective == NEF ) {
    if ( !n->lh || 
	 !n->rh ||
	 n->neft < 0 ) {
      return FALSE;
    }
    return TRUE;
  }

  return FALSE;

}



Bool make_conjunction_of_literals( PlNode **n )

{

  PlNode *tmp, *i;

  if ( !(*n) ) {
    return FALSE;
  }

  if ( (*n)->connective != AND ) {
    if ( (*n)->connective == NOT ) {
      if ( !((*n)->sons) ||
	   (*n)->sons->connective != ATOM ) {
	return FALSE;
      }
      tmp = new_PlNode( NOT );
      tmp->sons = (*n)->sons;
      (*n)->connective = AND;
      (*n)->sons = tmp;
      return TRUE;
    }
    if ( (*n)->connective == NEF ) {
      tmp = new_PlNode( NEF );
      tmp->neft = (*n)->neft;
      tmp->lh = (*n)->lh;
      tmp->rh = (*n)->rh;
      (*n)->lh = NULL;
      (*n)->rh = NULL;
      (*n)->neft = -1;
      (*n)->connective = AND;
      (*n)->sons = tmp;
      return TRUE;
    }
    if ( (*n)->connective != ATOM ) {
      return FALSE;
    }
    tmp = new_PlNode( ATOM );
    tmp->atom = (*n)->atom;
    (*n)->atom = NULL;
    (*n)->connective = AND;
    (*n)->sons = tmp;
    return TRUE;
  }

  for ( i = (*n)->sons; i; i = i->next ) {
    if ( !is_eff_literal( i ) ) {
      return FALSE;
    }
  }

  return TRUE;

}


