/**
 * @file    vsvr.h
 * @author  Thomas Lewiner <tomlew@mat.puc-rio.br>
 * @author  Math Dept, PUC-Rio
 * @version 2008.1
 * @date    07/03/2008
 *
 * @brief   Very Simple Volume Rendering
 */
//________________________________________________


#if !defined(WIN32) || defined(__CYGWIN__)
#pragma implementation
#endif // WIN32


#include "vsvr2.h"

#ifdef _DEBUG
#include <stdio.h>
#define PRINT_GL_DEBUG  { printf( "openGL watch at line %d: %s\n", __LINE__, ::gluErrorString( ::glGetError() ) ) ; }
#else  // _DEBUG
#define PRINT_GL_DEBUG  {}
#endif // _DEBUG


//_____________________________________________________________________________
// loads the texture and renders
bool VSVR2::gl_render( int nslices /*= tex_ni()*/ )
//-----------------------------------------------------------------------------
{
  if( !_tex || !_tf ) return false ;

 // if( _rescale ) tex_rescale() ;

  if( !tf_glload() || !tex_glload() )
  {
    printf( "could not load texture!\n" ) ;
    return false ;
  }

  return gl_redisplay( nslices ) ;
}
//_____________________________________________________________________________




//_____________________________________________________________________________
// redisplay with the current setting (possibly a different viewpoint)
bool VSVR2::gl_redisplay( int nslices /*= tex_ni()*/ ) 
//-----------------------------------------------------------------------------
{
  if( !_tex || !_tf ) return false ;
  if (_tex_changed) {
      tex_glreload();
	  _tex_changed = false;
  }
  // sets the openGL attributes and clipping planes
  gl_set () ;
  gl_clip() ;

  //--------------------------------------------------//
  // gets the direction of the observer
  double  gl_model[16] ; // = { 1.0f,0.0f,0.0f,0.0f, 0.0f,0.0f,-1.0f,0.0f, 0.0f,-1.0f,0.0f,0.0f, 0.0f,0.0f,0.0f,1.0f } ;
  double  gl_proj [16] ; // = { 1.0f,0.0f,0.0f,0.0f, 0.0f,1.0f,0.0f,0.0f, 0.0f,0.0f,1.0f,0.0f, 0.0f,0.0f,0.0f,1.0f } ;
  GLint     gl_view [ 4] ;
  glGetDoublev (GL_MODELVIEW_MATRIX , gl_model);
  glGetDoublev (GL_PROJECTION_MATRIX, gl_proj );
  glGetIntegerv(GL_VIEWPORT         , gl_view );



  //--------------------------------------------------//
  // gets the bounding box of the grid in the screen coordinates
  double xmin=FLT_MAX, xmax=-FLT_MAX, ymin=FLT_MAX, ymax=-FLT_MAX, zmin=FLT_MAX, zmax=-FLT_MAX;
  for( int i = 0; i < 8; ++i )
  {
    float bbx = (i&1) ? (float)tex_ni() : 0.0f ;
    float bby = (i&2) ? (float)tex_nj() : 0.0f ;
    float bbz = (i&4) ? (float)tex_nk() : 0.0f ;

    double x,y,z ;
    gluProject( bbx,bby,bbz, gl_model, gl_proj, gl_view, &x, &y, &z ) ;

    if( x < xmin ) xmin = x;
    if( x > xmax ) xmax = x;
    if( y < ymin ) ymin = y;
    if( y > ymax ) ymax = y;
    if( z < zmin ) zmin = z;
    if( z > zmax ) zmax = z;
  }

  //--------------------------------------------------//
  // world to tex coordinates
  double fx = 1.0 / tex_ni() ;
  double fy = 1.0 / tex_nj() ;
  double fz = 1.0 / tex_nk() ;

  //--------------------------------------------------//
  // draw each slice of the texture in the viewer coordinates
  float dz = (float)( (zmax-zmin) / nslices ) ;
  float z  = (float)zmax - dz/2.0f ;

  //glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  glBegin( GL_QUADS ) ;
  {
    for( int n = nslices-1 ; n >= 0 ; --n, z -= dz )
    {
      GLdouble point[3] ;
      gluUnProject( xmin,ymin,z, gl_model, gl_proj, gl_view, point + 0, point + 1, point + 2 ) ;
      glTexCoord3d( fx*point[0], fy*point[1], fz*point[2] );
      glVertex3dv( point ) ;

      gluUnProject( xmax,ymin,z, gl_model, gl_proj, gl_view, point + 0, point + 1, point + 2 ) ;
      glTexCoord3d( fx*point[0], fy*point[1], fz*point[2] );
      glVertex3dv( point ) ;

      gluUnProject( xmax,ymax,z, gl_model, gl_proj, gl_view, point + 0, point + 1, point + 2 ) ;
      glTexCoord3d( fx*point[0], fy*point[1], fz*point[2] );
      glVertex3dv( point ) ;

      gluUnProject( xmin,ymax,z, gl_model, gl_proj, gl_view, point + 0, point + 1, point + 2 ) ;
      glTexCoord3d( fx*point[0], fy*point[1], fz*point[2] );
      glVertex3dv( point ) ;
    }
  }
  glEnd() ; // GL_QUADS


  // unsets the openGL attributes and clipping planes
  gl_unclip() ;
  gl_unset () ;

  return true ;
}
//_____________________________________________________________________________




//_____________________________________________________________________________
//_____________________________________________________________________________




//_____________________________________________________________________________
// rescale the texels to match the transfer function size
void VSVR2::tex_rescale()
//-----------------------------------------------------------------------------
{
  // gets the maximal values
  float tex_min = FLT_MAX,  tex_max = -FLT_MAX ;
  int n = tex_ni()*tex_nj()*tex_nk() ;
  float *ptr = _tex ;
  /*for( int i = 0 ; i < n ; ++i, ++ptr )
  {
    float tex = *ptr ;
    if( tex_min > tex ) tex_min = tex ;
    if( tex_max < tex ) tex_max = tex ;
  }*/

  tex_min = _tf_min;
  tex_max = _tf_max;

  // rescale the values
  float tex_fact = (float)tf_size() / (tex_max - tex_min) ;
  if( tex_fact < FLT_EPSILON || tex_fact > 1e5 ) return ;
  ptr = _tex ;
  for( int i = 0 ; i < n ; ++i, ++ptr )
  {
    *ptr = (*ptr-tex_min) * tex_fact ;
  }

  _rescale = false ;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// loads the 3D texture
bool VSVR2::tex_glload()
//-----------------------------------------------------------------------------
{
		//printf("got herea \n");

  tex_glunload() ;

  // enable 3D texture
 // glEnable(GL_TEXTURE_3D);
		//printf("got herea1 \n");

  // init the 3D texture
  glGenTextures(1, &_tex_glid);
  glActiveTexture(GL_TEXTURE0 + 1);

  glBindTexture(GL_TEXTURE_3D, _tex_glid );
		//printf("got herea2 \n");

  // texture environment setup ( GL_CLAMP_TO_EDGE avoids invalid mapping at the texture border )
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  
  //printf("%d got herea3 = (%d, %d, %d) %d \n", glTexImage3D, (int) tex_ni(),     (int) tex_nj(),  (int) tex_nk(),_tex);

  // load the texture image
  glTexImage3D(GL_TEXTURE_3D,  // target
    0,                                  // level
    GL_RED/*GL_RGBA*/,                            // internal format
    (int) tex_ni(),                     // width
    (int) tex_nj(),                     // height
    (int) tex_nk(),                     // depth
    0,                                  // border
    GL_RED,                     // format
    GL_FLOAT,                           // type
    _tex            );                  // buffer
		//printf("got herea3.5 \n");

  //glPixelTransferi(GL_MAP_COLOR, GL_FALSE);
		//printf("got herea4 \n");

  //glDisable(GL_TEXTURE_3D);
	//printf("got here2 \n");
//
  return true;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// loads the transfer function
bool VSVR2::tf_glload() 
//-----------------------------------------------------------------------------
{

	//printf("got here \n");
	//glPixelTransferi(GL_MAP_COLOR  , GL_TRUE);

	//for (int i = 0;  i < 256; i++) {
	//	printf("%f %f %f %f \n", _tf[i*4 + 0],_tf[i*4 + 1],_tf[i*4 + 2],_tf[i*4 + 3]);
	//}
	//  float *ptr = _tf ;
  //glPixelMapfv(GL_PIXEL_MAP_I_TO_R, tf_size(), ptr);  ptr += tf_size() ;
 // glPixelMapfv(GL_PIXEL_MAP_I_TO_G, tf_size(), ptr);  ptr += tf_size() ;
  //glPixelMapfv(GL_PIXEL_MAP_I_TO_B, tf_size(), ptr);  ptr += tf_size() ;
  //glPixelMapfv(GL_PIXEL_MAP_I_TO_A, tf_size(), ptr);
	//printf("got here 1\n");
	tf_glunload();
	
	glGenTextures(1, &_tf_glid);
			glActiveTexture(GL_TEXTURE0 + 2);
	///glActiveTexture( _gtexture);
	//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glBindTexture(GL_TEXTURE_1D, _tf_glid);
	
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, TFSIZE,  0, GL_RGBA, GL_FLOAT, (const GLvoid *) _tf);



  return true;
}
//_____________________________________________________________________________

//_____________________________________________________________________________
// reloads the 3D texture with updated data
bool VSVR2::tex_glreload()
{
    // Bind the existing texture
    glBindTexture(GL_TEXTURE_3D, _tex_glid);

    // Update the texture data with new values from _tex
    glTexSubImage3D(GL_TEXTURE_3D,     // target
        0,                  // level
        0, 0, 0,            // offset (start at (0, 0, 0))
        (int)tex_ni(),     // width
        (int)tex_nj(),     // height
        (int)tex_nk(),     // depth
        GL_RED,             // format
        GL_FLOAT,           // type
        _tex);              // buffer with new texture data

// Optionally: Update texture parameters if needed (e.g., filtering)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    return true;
}


//_____________________________________________________________________________
// unloads the 3D texture
void VSVR2::tex_glunload()
//-----------------------------------------------------------------------------
{
  glDeleteTextures( 1, &_tex_glid ) ;
  _tex_glid = 0 ;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// unloads the transfer function
void VSVR2::tf_glunload() 
//-----------------------------------------------------------------------------
{
	glDeleteTextures( 1, &_tf_glid ) ;
	_tf_glid = 0 ;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// sets the openGL attributes
void VSVR2::gl_set() const
//-----------------------------------------------------------------------------
{
  // push the relevant parts of the OpenGL state
  glPushAttrib(GL_COLOR_BUFFER_BIT   |
                 GL_DEPTH_BUFFER_BIT   |
                 GL_ENABLE_BIT         |
                 GL_LIGHTING_BIT       |
                 GL_POLYGON_BIT        |
                 GL_TEXTURE_BIT);


  // openGL setup
  //glEnable (GL_TEXTURE_3D);
  //glBindTexture(GL_TEXTURE_3D, _tex_glid);
  glDisable(GL_LIGHTING);
  glDisable(GL_CULL_FACE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  	glUseProgramObjectARB(_sprog);
			//printf("glerror:gUseProgramObjectArb:%d\n", glGetError());

	GLint loc2 = glGetUniformLocationARB(_sprog, "voltex");
		if (loc2 == -1) {
			printf( "Warning: missing volume in frag shader: %d\n", glGetError() );
			
		} else {

			glActiveTexture(GL_TEXTURE0 + 1);
			glBindTexture(GL_TEXTURE_3D, _tex_glid); 
			glUniform1i(loc2, 1);
		}
	loc2 = glGetUniformLocationARB(_sprog, "transf");
		if (loc2 == -1) {
			printf( "Warning: missing tex in frag shader: %d\n", glGetError() );
			
		} else {

			glActiveTexture(GL_TEXTURE0 + 2);
			glBindTexture(GL_TEXTURE_1D, _tf_glid); 
			glUniform1i(loc2, 2);
		}


  // enable alpha blending
  glEnable   (GL_BLEND);
  //glDepthMask(GL_FALSE);
  glBlendEquation(GL_FUNC_ADD);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// unsets the openGL attributes
void VSVR2::gl_unset() const
//-----------------------------------------------------------------------------
{
	glUseProgramObjectARB(0);
	//glDisable (GL_TEXTURE_3D);
  
  glDisable(GL_BLEND);
  //glDepthMask(GL_TRUE);
  glPopAttrib() ;
}
//_____________________________________________________________________________




//_____________________________________________________________________________
// sets the clipping planes (uses the 6 first)
void VSVR2::gl_clip() const
//-----------------------------------------------------------------------------
{

  // clip the 6 faces of the cube
  GLdouble plane[4] ;
  plane[0] = +1. ;  plane[1] =  0. ;  plane[2] =  0. ;  plane[3] = FLT_EPSILON ;
  glEnable( GL_CLIP_PLANE0 ) ;
  glClipPlane( GL_CLIP_PLANE0, plane ) ;

  plane[0] = -1. ;  plane[1] =  0. ;  plane[2] =  0. ;  plane[3] = tex_ni() - FLT_EPSILON ;
  glEnable( GL_CLIP_PLANE1 ) ;
  glClipPlane( GL_CLIP_PLANE1, plane ) ;

  plane[0] =  0. ;  plane[1] = +1. ;  plane[2] =  0. ;  plane[3] = FLT_EPSILON ;
  glEnable( GL_CLIP_PLANE2 ) ;
  glClipPlane( GL_CLIP_PLANE2, plane ) ;

  plane[0] =  0. ;  plane[1] = -1. ;  plane[2] =  0. ;  plane[3] = tex_nj() + FLT_EPSILON ;
  glEnable( GL_CLIP_PLANE3 ) ;
  glClipPlane( GL_CLIP_PLANE3, plane ) ;

  plane[0] =  0. ;  plane[1] =  0. ;  plane[2] = +1. ;  plane[3] = FLT_EPSILON ;
  glEnable( GL_CLIP_PLANE4 ) ;
  glClipPlane( GL_CLIP_PLANE4, plane ) ;

  plane[0] =  0. ;  plane[1] =  0. ;  plane[2] = -1. ;  plane[3] = tex_nk() + FLT_EPSILON ;
  glEnable( GL_CLIP_PLANE5 ) ;
  glClipPlane( GL_CLIP_PLANE5, plane ) ;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// unsets the clipping planes
void VSVR2::gl_unclip() const
//-----------------------------------------------------------------------------
{
  // disable cube clip plane
  glDisable( GL_CLIP_PLANE0 ) ;
  glDisable( GL_CLIP_PLANE1 ) ;
  glDisable( GL_CLIP_PLANE2 ) ;
  glDisable( GL_CLIP_PLANE3 ) ;
  glDisable( GL_CLIP_PLANE4 ) ;
  glDisable( GL_CLIP_PLANE5 ) ;
}
//_____________________________________________________________________________
